"""The DAT-file and population-analysis processors reach the materials database.

process_calculation_dat_files and process_material_population_analysis
imported MaterialDatabase from ``material_database``, a module that no longer
exists (it is mace/database/materials.py), so both stopped with
ModuleNotFoundError before doing anything.
"""
import json

from mace.database.materials import MaterialDatabase

TINY_BAND_DAT = """\
# NKPT 3 NBND 2 NSPIN 1
# NPANEL 1
#   0   G
#   2   X
 0.000  -0.200  0.100
 0.500  -0.150  0.150
 1.000  -0.100  0.200
"""


def _db(tmp_path, monkeypatch):
    # MaterialDatabase also opens structures.db in the working directory.
    monkeypatch.chdir(tmp_path)
    db_path = str(tmp_path / "materials.db")
    db = MaterialDatabase(db_path)
    db.create_material("m1", "C2")
    calc_id = db.create_calculation("m1", "BAND")
    return db, db_path, calc_id


def test_dat_files_are_processed_and_saved(tmp_path, monkeypatch):
    from mace.utils.dat_file_processor import process_calculation_dat_files

    db, db_path, calc_id = _db(tmp_path, monkeypatch)
    work = tmp_path / "work"
    work.mkdir()
    (work / "m1.BAND.DAT").write_text(TINY_BAND_DAT)

    results = process_calculation_dat_files(calc_id, str(work), db_path)

    assert list(results) == ["band_structure_m1.BAND"]
    assert "error" not in results["band_structure_m1.BAND"]
    with db._get_connection() as conn:
        rows = conn.execute(
            "SELECT property_name FROM properties WHERE calc_id = ?", (calc_id,)
        ).fetchall()
    assert [r[0] for r in rows] == ["dat_file_band_structure_m1.BAND"]


def test_population_analysis_is_read_from_the_database(tmp_path, monkeypatch):
    from mace.utils.population_analysis_processor import (
        process_material_population_analysis)

    db, db_path, calc_id = _db(tmp_path, monkeypatch)
    mulliken = {"atoms": [
        {"atom_number": 1, "element": "C", "atomic_number": 6, "mulliken_charge": 6.1},
        {"atom_number": 2, "element": "C", "atomic_number": 6, "mulliken_charge": 5.9},
    ]}
    with db._get_connection() as conn:
        conn.execute(
            "INSERT INTO properties (material_id, calc_id, property_category, "
            "property_name, property_value_text, extracted_at) "
            "VALUES (?, ?, 'population_analysis', 'mulliken_population', ?, datetime('now'))",
            ("m1", calc_id, json.dumps(mulliken)))

    result = process_material_population_analysis("m1", db_path)

    charges = result["atomic_charges"]
    assert [c["atom_number"] for c in charges] == [1, 2]
    assert [round(c["mulliken_charge"], 6) for c in charges] == [-0.1, 0.1]
