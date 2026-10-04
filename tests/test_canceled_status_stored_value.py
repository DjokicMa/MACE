"""A canceled job keeps its original stored status, whatever MACE prints.

The American-spelling sweep changed what MACE prints ("Canceled job ...") but
not the status value in the calculations table (British spelling, pinned in
STORED below): existing databases already hold it, and the cleanup and
completion code compare against it.
"""

import sqlite3
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mace.database.materials import STATUS_CANCELED_DB, MaterialDatabase  # noqa: E402

STORED = "cancelled"  # british-ok: the stored database value this test pins


def _db(tmp_path):
    db = MaterialDatabase(str(tmp_path / "materials.db"))
    db.create_material("si", "Si")
    return db


def _row(tmp_path, calc_id):
    with sqlite3.connect(tmp_path / "materials.db") as con:
        return con.execute("SELECT status, completed_at FROM calculations WHERE calc_id=?",
                           (calc_id,)).fetchone()


def test_stored_value_keeps_its_original_spelling():
    assert STATUS_CANCELED_DB == STORED


def test_canceling_a_calculation_stores_the_old_value_and_completes_it(tmp_path):
    db = _db(tmp_path)
    calc_id = db.create_calculation("si", "OPT")
    db.update_calculation_status(calc_id, STATUS_CANCELED_DB)
    status, completed_at = _row(tmp_path, calc_id)
    assert status == STORED
    assert completed_at


def test_cleanup_removes_old_rows_written_by_earlier_versions(tmp_path):
    db = _db(tmp_path)
    old = db.create_calculation("si", "OPT")
    keep = db.create_calculation("si", "SP")
    long_ago = (datetime.now() - timedelta(days=90)).isoformat()
    with sqlite3.connect(tmp_path / "materials.db") as con:
        con.execute("UPDATE calculations SET status=?, created_at=? WHERE calc_id=?",
                    (STORED, long_ago, old))
        con.execute("UPDATE calculations SET created_at=? WHERE calc_id=?", (long_ago, keep))
    db.cleanup_old_records(days_old=30)
    assert _row(tmp_path, old) is None
    assert _row(tmp_path, keep) is not None


def test_queue_manager_cancel_job_stores_the_old_value(tmp_path, monkeypatch, capsys):
    from mace.queue import manager as M

    db = _db(tmp_path)
    calc_id = db.create_calculation("si", "OPT")
    monkeypatch.setattr(M.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    fake = types.SimpleNamespace(db=db, enable_tracking=True)
    M.EnhancedCrystalQueueManager.cancel_job(fake, "4242", calc_id)
    assert _row(tmp_path, calc_id)[0] == STORED
    assert "Canceled job 4242" in capsys.readouterr().out
