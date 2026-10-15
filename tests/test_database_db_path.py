"""`mace database --db-path PATH` uses PATH.

The option was listed but not read: every action opened the context's (or the
current folder's) materials.db. Now --db-path is honored by every action, and
a missing file is an error (no action creates a database), not a new empty
materials.db. Real CLI subprocess."""
import os
import sqlite3
import subprocess
import sys

from conftest import REPO_ROOT


def _env():
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _db(cwd, *args):
    return subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "database", *args],
                          capture_output=True, text=True, env=_env(), cwd=str(cwd), timeout=300)


def test_db_path_is_used(tmp_path):
    from mace.database.materials import MaterialDatabase
    other = tmp_path / "elsewhere" / "mine.db"
    other.parent.mkdir()
    MaterialDatabase(str(other))                     # the schema
    c = sqlite3.connect(str(other))
    c.execute("INSERT INTO materials (material_id, formula, created_at, updated_at) "
              "VALUES ('mat_x', 'C2', '2026-01-01', '2026-01-01')")
    c.commit()
    c.close()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    p = _db(cwd, "--action", "query", "--db-path", str(other))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "mat_x" in p.stdout
    assert not (cwd / "materials.db").exists()


def test_missing_db_path_is_an_error(tmp_path):
    p = _db(tmp_path, "--action", "stats", "--db-path", str(tmp_path / "nope.db"))
    assert p.returncode != 0
    assert "nope.db" in p.stdout + p.stderr
    assert not (tmp_path / "nope.db").exists() and not (tmp_path / "materials.db").exists()
