"""The 1.1.6 properties schema and the migration of a database an earlier
MACE built (MaterialDatabase._migrate_properties_116): duplicates removed
(the newest row kept), a unique (material, calculation, property) index, an
index on calc_id, the large JSON values moved to property_blobs, a backup
first, once (PRAGMA user_version), under the write lock - and the database
still readable the way MACE 1.1.5 reads it."""
import contextlib
import json
import sqlite3
from pathlib import Path

import pytest

from mace.database import dbsafety
from mace.database import materials as M
from mace.database.materials import MaterialDatabase


def _as_115(path: Path):
    """Turn a fresh database into one MACE 1.1.5 left: no unique index, no
    calc_id index, no property_blobs, user_version 0."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        DROP INDEX IF EXISTS idx_properties_key;
        DROP INDEX IF EXISTS idx_properties_calc;
        DROP TABLE IF EXISTS property_blobs;
        PRAGMA user_version = 0;
    """)
    conn.commit()
    conn.close()


def _row(conn, mid, cid, name, value=None, text=None, at="2026-01-01T00:00:00", cat="electronic"):
    conn.execute("INSERT INTO properties (material_id, calc_id, property_category, property_name, "
                 "property_value, property_value_text, property_unit, extracted_at, extractor_script) "
                 "VALUES (?, ?, ?, ?, ?, ?, 'eV', ?, 'crystal_property_extractor.py')",
                 (mid, cid, cat, name, value, text, at))


@pytest.fixture
def old_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # structures.db
    path = tmp_path / "materials.db"
    db = MaterialDatabase(str(path))
    db.create_material("m1", "C2")
    cid = db.create_calculation("m1", "SP")
    del db
    _as_115(path)
    conn = sqlite3.connect(path)
    # a duplicate set (the bug 1.1.5 fixed for new rows; old databases keep them)
    _row(conn, "m1", cid, "band_gap", 1.0, "1.0", at="2026-01-01T00:00:00")
    _row(conn, "m1", cid, "band_gap", 2.0, "2.0", at="2026-03-01T00:00:00")   # newest: kept
    _row(conn, "m1", cid, "band_gap", 1.5, "1.5", at="2026-02-01T00:00:00")
    # untracked rows (calc_id NULL) are one set per material too
    _row(conn, "m1", None, "total_energy_au", -1.0, "-1.0", at="2026-01-01T00:00:00")
    _row(conn, "m1", None, "total_energy_au", -2.0, "-2.0", at="2026-01-01T00:00:00")  # same time: higher id
    # a large JSON value and an old name
    _row(conn, "m1", cid, "neighbor_analysis", None, json.dumps([{"atom": 1}]), cat="structural")
    _row(conn, "m1", cid, "total_cpu_time", 12.5, "12.5", cat="computational")
    conn.commit()
    conn.close()
    return path, cid


def test_migration_dedups_indexes_moves_blobs_and_backs_up(old_db, capsys):
    path, cid = old_db
    db = MaterialDatabase(str(path))
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == M.SCHEMA_VERSION
    rows = conn.execute("SELECT property_name, calc_id, property_value FROM properties "
                        "ORDER BY property_name").fetchall()
    assert ("band_gap", cid, 2.0) in rows and len([r for r in rows if r[0] == "band_gap"]) == 1
    assert [r[2] for r in rows if r[0] == "total_energy_au"] == [-2.0]
    assert not [r for r in rows if r[0] == "neighbor_analysis"]
    blobs = conn.execute("SELECT property_name, calc_id, property_value_text FROM property_blobs").fetchall()
    assert blobs == [("neighbor_analysis", cid, json.dumps([{"atom": 1}]))]
    idx = {r[1]: r[2] for r in conn.execute("PRAGMA index_list(properties)")}
    assert idx.get("idx_properties_key") == 1 and "idx_properties_calc" in idx
    with pytest.raises(sqlite3.IntegrityError):
        _row(conn, "m1", cid, "band_gap", 3.0)
    with pytest.raises(sqlite3.IntegrityError):
        _row(conn, "m1", None, "total_energy_au", 3.0)   # NULL calc_id is one key too
    conn.close()
    # the backup is the database as it was, beside it, and no MACE backup pruning matches it
    backups = sorted(path.parent.glob("materials_pre116_backup_*.db"))
    assert len(backups) == 1 and not dbsafety.BACKUP_NAME.match(backups[0].name)
    b = sqlite3.connect(backups[0])
    assert b.execute("SELECT count(*) FROM properties WHERE property_name = 'band_gap'").fetchone()[0] == 3
    assert b.execute("PRAGMA user_version").fetchone()[0] == 0
    b.close()
    err = capsys.readouterr().err
    assert "MACE 1.1.6 schema" in err and "3 duplicate property rows removed" in err
    # the read paths: the JSON value with the rest, the old name as its new one
    props = {p["property_name"]: p for p in db.get_material_properties("m1")}
    assert "neighbor_analysis" in props and props["cpu_time_total_s"]["property_value"] == 12.5
    assert "neighbor_analysis" not in {p["property_name"] for p in db.get_all_properties()}


def test_migration_runs_once(old_db):
    path, _ = old_db
    MaterialDatabase(str(path))
    MaterialDatabase(str(path))
    assert len(list(path.parent.glob("materials_pre116_backup_*.db"))) == 1
    assert MaterialDatabase(str(path))._migrate_properties_116() is False


def test_a_new_database_needs_no_backup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "fresh.db"
    MaterialDatabase(str(path))
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == M.SCHEMA_VERSION
    assert "idx_properties_key" in {r[1] for r in conn.execute("PRAGMA index_list(properties)")}
    assert not list(tmp_path.glob("*pre116*"))


def test_a_failed_backup_changes_nothing(old_db, monkeypatch, capsys):
    path, _ = old_db

    def fail(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(dbsafety, "snapshot_backup", fail)
    db = MaterialDatabase(str(path))
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM properties WHERE property_name = 'band_gap'").fetchone()[0] == 3
    conn.close()
    assert "its backup failed (disk full)" in capsys.readouterr().err
    # MACE works on without the new index; the next open migrates
    assert db.get_material_properties("m1")
    monkeypatch.undo()
    MaterialDatabase(str(path))
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == M.SCHEMA_VERSION


def test_the_migration_takes_the_write_lock(old_db, monkeypatch):
    path, _ = old_db
    seen = []
    real = dbsafety.write_guard

    @contextlib.contextmanager
    def guard(p):
        seen.append(Path(p).name)
        with real(p):
            yield
    monkeypatch.setattr(dbsafety, "write_guard", guard)
    db = MaterialDatabase(str(path), auto_initialize=False)
    db._initialize_database()
    calls = len(seen)
    assert calls and set(seen) == {"materials.db"}
    seen.clear()
    db2 = MaterialDatabase(str(path), auto_initialize=False)
    _as_115(path)
    db2._migrate_properties_116()
    assert seen, "the migration's writes did not go through dbsafety.write_guard"


def test_mace_115_reads_a_migrated_database(old_db):
    """The queries MACE 1.1.5's MaterialDatabase and its GUI run still work
    (the confidence column is kept; the JSON values moved to property_blobs
    are what 1.1.5 does not see)."""
    path, cid = old_db
    MaterialDatabase(str(path))
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = conn.execute("""
        SELECT property_id, material_id, calc_id, property_category,
               property_name, property_value, property_value_text,
               property_unit, confidence, extractor_script, extracted_at
        FROM properties WHERE material_id = ? ORDER BY property_category, property_name""", ("m1",)).fetchall()
    assert {r[4] for r in rows} == {"band_gap", "total_energy_au", "total_cpu_time"}
    assert conn.execute("SELECT DISTINCT property_name FROM properties ORDER BY property_name").fetchall()
    conn.close()


def test_store_material_property_replaces_its_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = MaterialDatabase(str(tmp_path / "m.db"))
    db.create_material("m1", "C2")
    db.store_material_property("m1", "band_gap", 1.0, "electronic")
    db.store_material_property("m1", "band_gap", 2.0, "electronic")
    assert [p["property_value"] for p in db.get_material_properties("m1")] == [2.0]


def test_dedupe_is_one_pass_on_a_large_database(tmp_path, monkeypatch):
    """The duplicate removal was quadratic in the rows of a property name (90 s
    for 215 k rows); one pass takes well under a second for 120 k."""
    import time

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "materials.db"
    db = MaterialDatabase(str(path))
    db.create_material("m1", "C2")
    del db
    _as_115(path)
    conn = sqlite3.connect(path)
    rows = [(f"m{i % 400}", f"c{i % 400}", "electronic", f"p{i // 400 % 100}", float(i), str(i),
             "eV", f"2026-01-0{1 + i // 40000}T00:00:00", "x") for i in range(120000)]
    conn.executemany("INSERT INTO properties (material_id, calc_id, property_category, "
                     "property_name, property_value, property_value_text, property_unit, "
                     "extracted_at, extractor_script) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    t0 = time.monotonic()
    MaterialDatabase(str(path))
    elapsed = time.monotonic() - t0
    conn = sqlite3.connect(path)
    kept = conn.execute("SELECT count(*), min(property_value) FROM properties").fetchone()
    conn.close()
    assert kept == (40000, 80000.0)   # the newest of each (material, calc, property) triple
    assert elapsed < 15, elapsed


def test_concurrent_first_opens_write_one_backup(old_db):
    """Each process opening a 1.1.5 database at the same time used to write
    its own full backup before migrating; the backup is now taken under the
    migration's write lock, so the others wait and find it migrated."""
    import subprocess
    import sys
    from conftest import REPO_ROOT
    path, _ = old_db
    # a slow backup (a large database): the openers overlap
    code = ("import sys, time; sys.path.insert(0, sys.argv[2]); "
            "from mace.database import dbsafety as D; f = D.snapshot_backup; "
            "D.snapshot_backup = lambda *a, **k: (time.sleep(1), f(*a, **k))[1]; "
            "from mace.database.materials import MaterialDatabase as M; M(sys.argv[1])")
    procs = [subprocess.Popen([sys.executable, "-c", code, str(path), str(REPO_ROOT)], cwd=str(path.parent),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(4)]
    assert [p.wait(timeout=120) for p in procs] == [0] * 4
    assert len(list(path.parent.glob("materials_pre116_backup_*.db"))) == 1
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == M.SCHEMA_VERSION
    conn.close()
