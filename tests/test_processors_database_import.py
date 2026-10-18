"""The population-analysis processor reaches the materials database.

process_material_population_analysis imported MaterialDatabase from
``material_database``, a module that no longer exists (it is
mace/database/materials.py), so it stopped with ModuleNotFoundError before
doing anything. (dat_file_processor.process_calculation_dat_files, which had
the same import, had no caller and is removed in 1.1.6.)
"""
import json

from mace.database.materials import MaterialDatabase

def _db(tmp_path, monkeypatch):
    # MaterialDatabase also opens structures.db in the working directory.
    monkeypatch.chdir(tmp_path)
    db_path = str(tmp_path / "materials.db")
    db = MaterialDatabase(db_path)
    db.create_material("m1", "C2")
    calc_id = db.create_calculation("m1", "BAND")
    return db, db_path, calc_id


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
