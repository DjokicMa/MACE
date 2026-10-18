"""The GUI's database reads while jobs write the same materials.db
(PLAN_analysis C17): the open rule by SQLite error code (never immutable
over a hot journal, never on a busy database), checked immutable reads,
503 busy answers that the page rides out, and the live refresh.

Databases are real SQLite files made with MaterialDatabase in tmp; writers
are subprocesses (the job callback's own API)."""
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

sys.path[:0] = [p for p in (str(H.REPO_ROOT),) if p not in sys.path]


def _env(**extra):
    e = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH",
                                                        "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e.update(PYTHONPATH=str(H.REPO_ROOT), OMP_NUM_THREADS="1", **extra)
    return e


MAKE = textwrap.dedent("""
    import sys
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(sys.argv[1], ase_db_path=sys.argv[1] + '.structures.db')
    n = int(sys.argv[2]); start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    for i in range(start, start + n):
        mid = f'mat{i:04d}'
        db.create_material(mid, 'C2', space_group=227)
        cid = db.create_calculation(material_id=mid, calc_type='SP', input_file=f'/x/{mid}.d12', work_dir='/x')
        db.update_calculation_status(cid, 'completed')
    """)


def make_db(path: Path, n: int, start: int = 0, **env):
    r = subprocess.run([sys.executable, "-c", MAKE, str(path), str(n), str(start)], capture_output=True, text=True,
                       env=_env(**env), timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def _gui_db():
    from mace.gui import db as gui_db
    return gui_db


# ------------------------------------------------------------------ C17-a
class _Err(sqlite3.OperationalError):
    def __init__(self, msg, code):
        super().__init__(msg)
        self.sqlite_errorcode = code


def test_open_rule_by_error_code(tmp_path, monkeypatch):
    gui_db = _gui_db()
    db = tmp_path / "materials.db"
    make_db(db, 2)
    monkeypatch.setattr(gui_db, "OPEN_BUSY_S", 0.3)
    calls = []

    def busy(uri):
        calls.append(uri)
        raise _Err("database is locked", 5)
    monkeypatch.setattr(gui_db, "_open", busy)
    with pytest.raises(gui_db.DbBusy) as e:
        gui_db.connect(db)
    assert e.value.reason == "busy" and not any("immutable" in c for c in calls) and len(calls) > 1

    def hot(uri):
        calls.append(uri)
        raise _Err("attempt to write a readonly database", 776)
    calls.clear()
    monkeypatch.setattr(gui_db, "_open", hot)
    with pytest.raises(gui_db.DbBusy) as e:
        gui_db.connect(db)
    assert e.value.reason == "interrupted" and len(calls) == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_read_only_folder_reads_a_snapshot(tmp_path):
    gui_db = _gui_db()
    d = tmp_path / "ro"
    d.mkdir()
    db = d / "materials.db"
    make_db(db, 3)
    c = sqlite3.connect(str(db))
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    c.close()
    d.chmod(0o555)
    try:
        page = gui_db.page(db, with_total=True)
        assert page["readonly_snapshot"] is True and page["total"] == 3
        d.chmod(0o755)
        (d / "materials.db-journal").write_bytes(b"\0" * 512)        # a writer's journal appears
        d.chmod(0o555)
        with pytest.raises(gui_db.DbBusy) as e:
            gui_db.page(db)
        assert e.value.reason == "interrupted"
    finally:
        d.chmod(0o755)


# ------------------------------------------------------------------ C17-b
@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_checked_read_retries_then_reports_busy(tmp_path):
    gui_db = _gui_db()
    d = tmp_path / "ro"
    d.mkdir()
    db = d / "materials.db"
    make_db(db, 2)
    d.chmod(0o555)
    try:
        n = [0]

        def changing(conn):
            n[0] += 1
            t = time.time() + n[0]
            os.utime(db, (t, t))                       # the file changes while the query runs
            return conn.execute("SELECT count(*) FROM materials").fetchone()[0]
        with pytest.raises(gui_db.DbBusy):
            gui_db.read(db, changing)
        assert n[0] == 2                               # one retry, then busy (never a 500)
    finally:
        d.chmod(0o755)


# ------------------------------------------------------------------ C17-e (real kill, hot journal)
KILLED = textwrap.dedent("""
    import sqlite3, sys, os
    c = sqlite3.connect(sys.argv[1], isolation_level=None)
    c.execute('PRAGMA journal_mode=DELETE')
    c.execute('PRAGMA cache_size=5')            # spill to the database file mid-transaction
    c.execute('BEGIN IMMEDIATE')
    c.execute("UPDATE materials SET formula = 'X'")
    c.execute("CREATE TABLE pad AS WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 4000) "
              "SELECT i, hex(randomblob(250)) AS b FROM n")
    print('mid', flush=True)
    import time; time.sleep(60)
    """)


def test_killed_writer_hot_journal_then_repaired(tmp_path):
    gui_db = _gui_db()
    db = tmp_path / "materials.db"
    make_db(db, 50, MACE_DB_JOURNAL="delete")
    c = sqlite3.connect(str(db))
    c.execute("PRAGMA journal_mode=DELETE")
    c.close()
    p = subprocess.Popen([sys.executable, "-c", KILLED, str(db)], stdout=subprocess.PIPE, text=True, env=_env())
    assert p.stdout.readline().strip() == "mid"
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    assert Path(str(db) + "-journal").exists()
    with pytest.raises(gui_db.DbBusy) as e:
        gui_db.connect(db)
    assert e.value.reason == "interrupted"
    # the next writer (MaterialDatabase, as the callback opens it) rolls the hot journal back
    make_db(db, 1, 50, MACE_DB_JOURNAL="delete")
    page = gui_db.page(db, with_total=True)
    assert page["total"] == 51 and not page["readonly_snapshot"]
    assert all(not str(r["formula"]).startswith("X") for r in page["rows"])


# ------------------------------------------------------------------ C17.5
def test_the_gui_never_opens_a_database_read_write(tmp_path, monkeypatch):
    """Database reads and the GUI's backups never instantiate MaterialDatabase."""
    db = tmp_path / "materials.db"
    make_db(db, 3)
    from mace.database import materials

    def refuse(*a, **k):
        raise AssertionError("MaterialDatabase was instantiated in the GUI process")
    monkeypatch.setattr(materials.MaterialDatabase, "__init__", refuse)
    gui_db = _gui_db()
    gui_db.page(db, with_total=True)
    gui_db.stats(db)
    gui_db.material(db, "mat0000")
    from mace.gui.dbsafety_loader import dbsafety
    b = dbsafety().snapshot_backup(db, tmp_path / "backups")
    assert b.is_file()


# ------------------------------------------------------------------ C17-c (e2e)
@pytest.fixture
def served(tmp_path):
    pytest.importorskip("selenium.webdriver")
    root = H.examples_copy(tmp_path)
    db = root / "live.db"
    make_db(db, 5, MACE_DB_JOURNAL="delete")
    app = U.start_app(tmp_path, root=root, MACE_DB_JOURNAL="delete")
    try:
        yield app, db
    finally:
        U.stop_app(app)


def _open_db(drv, rel):
    from selenium.common.exceptions import StaleElementReferenceException
    for attempt in range(5):         # the list is drawn again once after the file details load
        try:
            row = drv.find_element("css selector", f'#file-list .file[data-path="{rel}"]')
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", row)
            row.click()
            break
        except StaleElementReferenceException:
            if attempt == 4:
                raise
            time.sleep(0.5)
    H.wait(drv, "return document.querySelectorAll('#db-host .dbv-tr[data-id]').length > 0", 30)


@pytest.mark.gui
def test_live_refresh_and_busy_chip(served):
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import Select
    (drv, base, root, xdg, proc, downloads), db = served
    # Settings > Files and data > Refresh the Database view: every 15 s (real keys and clicks)
    drv.execute_script("document.body.focus()")
    U.keys(drv, Keys.CONTROL, ",")
    H.wait(drv, "return document.querySelector('#settings') && document.querySelector('#settings').open")
    drv.find_element("css selector", "#settings [data-sec='files']").click()
    Select(drv.find_element("css selector", "#set-database-refreshSeconds")).select_by_value("15")
    U.keys(drv, Keys.ESCAPE)
    assert drv.execute_script("return window.__mace.settings.get('database.refreshSeconds')") == 15
    _open_db(drv, "live.db")
    # sort by formula descending and select a row: both must survive the refresh
    drv.find_element("css selector", "#db-host [data-sort='material_id']").click()
    H.wait(drv, "return document.querySelector('#db-host .dbv-th[aria-sort=descending]')", 10)
    # the header turns at once; the rows are placeholders until the sorted first page arrives
    H.wait(drv, "const r = document.querySelector('#db-host .dbv-tr[data-id]'); return r && r.dataset.id === 'mat0004'", 10)
    row = drv.find_element("css selector", "#db-host .dbv-tr[data-id='mat0002']")
    row.click()
    H.wait(drv, "return document.querySelector('#db-host .dbv-tr.sel[data-id=mat0002]')", 10)
    make_db(db, 3, 5, MACE_DB_JOURNAL="delete")                       # a burst of 3 new calculations
    H.wait(drv, "return /Updated .* · 3 new calculations/.test(document.querySelector('#db-host .dbv-live').textContent)", 40)
    assert drv.execute_script("return document.querySelector('#db-host .dbv-th[aria-sort=descending] [data-sort]').dataset.sort") == "material_id"
    assert drv.execute_script("return !!document.querySelector('#db-host .dbv-tr.sel[data-id=mat0002]')")
    assert drv.execute_script("return document.querySelector('#db-host .dbv-tr[data-id]').dataset.id") == "mat0007"
    # an exclusive lock for 8 s: the status line says busy, then clears by itself
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent("""
        import sqlite3, sys, time
        c = sqlite3.connect(sys.argv[1], isolation_level=None)
        c.execute('BEGIN EXCLUSIVE'); print('held', flush=True); time.sleep(8); c.execute('ROLLBACK')
        """), str(db)], stdout=subprocess.PIPE, text=True, env=_env())
    try:
        assert holder.stdout.readline().strip() == "held"
        drv.find_element("css selector", "#db-host [data-act=refresh]").click()
        H.wait(drv, "return /Database busy/.test(document.querySelector('#db-host .dbv-live').textContent)", 15)
        H.wait(drv, "return /Updated/.test(document.querySelector('#db-host .dbv-live').textContent)", 40)
    finally:
        holder.wait(timeout=30)
    assert not U.toasts(drv) or all("busy" not in t.lower() for t in U.toasts(drv))
    U.no_errors(drv)
    log = U.req(base, "api/test/dblog")[2]
    assert log["immutable_opens"] == []


@pytest.mark.gui
@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_read_only_folder_banner_and_refresh(tmp_path):
    pytest.importorskip("selenium.webdriver")
    root = H.examples_copy(tmp_path)
    sub = root / "ro"
    sub.mkdir()
    db = sub / "materials.db"
    make_db(db, 4)
    sub.chmod(0o555)
    app = U.start_app(tmp_path, root=root)
    try:
        drv = app[0]
        _open_db(drv, "ro/materials.db")
        H.wait(drv, "return /Read-only snapshot/.test(document.querySelector('#db-host .dbv-note').textContent)", 10)
        sub.chmod(0o755)
        make_db(db, 2, 4)                                   # an out-of-band write by the owner
        sub.chmod(0o555)
        drv.find_element("css selector", "#db-host [data-act=refresh]").click()
        H.wait(drv, "return /6 materials/.test(document.querySelector('#db-host .dbv-count').textContent)", 20)
        U.no_errors(drv)
    finally:
        sub.chmod(0o755)
        U.stop_app(app)
