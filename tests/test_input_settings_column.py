"""The extracted D12/D3 input settings live in calculations.input_settings_json.

settings_extractor wrote them there, but query_calculation_settings and the
`queries.py` input-settings commands read settings_json, which holds the
job's own metadata (workflow_id, output_file, ...). A stored extraction was
never shown, and a calculation with only workflow metadata was listed as
having input settings. Databases written before the column existed kept the
extraction in settings_json; those rows must still be found.
"""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from conftest import REPO_ROOT

# A minimal hand-written deck (generic diamond), enough for the extractor.
DECK = """Diamond
CRYSTAL
0 0 0
227
3.567
1
6 0.0 0.0 0.0
OPTGEOM
ENDOPT
END
6 0
END
DFT
PBE
END
SHRINK
8 8
TOLINTEG
7 7 7 7 14
END
"""


def _db(tmp_path):
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(db_path=str(tmp_path / "materials.db"),
                          ase_db_path=str(tmp_path / "structures.db"))
    db.create_material(material_id="dia", formula="C")
    return db


def _run_queries(tmp_path, *args):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "MACE_HOME")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "mace.database.query.queries",
         "--db-path", str(tmp_path / "materials.db"), *args],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_extracted_settings_are_returned_not_the_job_metadata(tmp_path):
    from mace.utils.settings_extractor import (
        extract_and_store_input_settings, query_calculation_settings)
    db = _db(tmp_path)
    deck = tmp_path / "dia.d12"
    deck.write_text(DECK)
    calc_id = db.create_calculation("dia", "OPT", input_file=str(deck),
                                    settings={"workflow_id": "wf_x"})
    assert extract_and_store_input_settings(calc_id, deck,
                                            str(tmp_path / "materials.db"))

    settings = query_calculation_settings(calc_id, str(tmp_path / "materials.db"))
    assert settings is not None
    assert "OPTGEOM" in settings["crystal_keywords"]
    assert "workflow_id" not in settings

    out = _run_queries(tmp_path, "--calc-id", calc_id)
    assert "OPTGEOM" in out
    out = _run_queries(tmp_path, "--material-id", "dia")
    assert "Keywords: CRYSTAL, OPTGEOM" in out
    out = _run_queries(tmp_path, "--settings-summary")
    assert "TOLINTEG" in out


def test_job_metadata_alone_is_not_input_settings(tmp_path):
    from mace.utils.settings_extractor import query_calculation_settings
    db = _db(tmp_path)
    calc_id = db.create_calculation("dia", "OPT", input_file="dia.d12",
                                    settings={"workflow_id": "wf_x"})
    assert query_calculation_settings(calc_id, str(tmp_path / "materials.db")) is None
    out = _run_queries(tmp_path, "--list-all")
    assert calc_id[:25] not in out
    assert "No calculations with input settings found" in out


def test_pre_column_database_extraction_in_settings_json_still_read(tmp_path):
    """Before input_settings_json existed the extractor wrote settings_json."""
    from mace.utils.settings_extractor import (
        extract_input_settings, query_calculation_settings)
    db = _db(tmp_path)
    deck = tmp_path / "dia.d12"
    deck.write_text(DECK)
    legacy = extract_input_settings(deck)
    calc_id = db.create_calculation("dia", "OPT", input_file=str(deck),
                                    settings=legacy)
    with sqlite3.connect(tmp_path / "materials.db") as conn:
        assert conn.execute("SELECT input_settings_json FROM calculations "
                            "WHERE calc_id = ?", (calc_id,)).fetchone()[0] is None

    settings = query_calculation_settings(calc_id, str(tmp_path / "materials.db"))
    assert settings is not None and "OPTGEOM" in settings["crystal_keywords"]
    out = _run_queries(tmp_path, "--material-id", "dia")
    assert "Keywords: CRYSTAL, OPTGEOM" in out
    out = _run_queries(tmp_path, "--list-all")
    assert calc_id[:25] in out
