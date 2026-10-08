"""mace/database/dbsafety.py: journal mode by file system, lock support from
the mount options, retries by SQLite error code, consistent backups and
the degraded-mode writer lock (PLAN_analysis C14).

Mount tables are synthetic mountinfo lines (no real host names or user
paths); databases are real SQLite files in tmp_path."""
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [p for p in (str(REPO),) if p not in sys.path]

from mace.database import dbsafety as D  # noqa: E402


def _env(**extra):
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e.update(PYTHONPATH=str(REPO), OMP_NUM_THREADS="1", **extra)
    return e


def _mi(fstype, mopts="rw,relatime", sopts="rw", mp="/data"):
    return (f"1 0 0:1 / / rw,relatime shared:1 - ext4 /dev/root rw\n"
            f"36 1 0:40 / {mp} {mopts} shared:20 - {fstype} server:/export {sopts}\n")


CASES = [
    ("ext4", "rw", "rw", "wal", "local"), ("xfs", "rw", "rw", "wal", "local"), ("btrfs", "rw", "rw", "wal", "local"),
    ("zfs", "rw", "rw", "wal", "local"), ("tmpfs", "rw", "rw", "wal", "local"),
    ("nfs", "rw", "rw,vers=3,hard,local_lock=none", "delete", "reliable"),
    ("nfs4", "rw", "rw,vers=4.2", "delete", "reliable"),
    ("gpfs", "rw", "rw", "delete", "reliable"),
    ("lustre", "rw,flock", "rw,flock", "delete", "reliable"),  # british-ok: a file system type
    ("cephfs", "rw", "rw", "delete", "reliable"),
    ("cifs", "rw", "rw", "delete", "reliable"),
    ("lustre", "rw,localflock", "rw,localflock", "delete", "unreliable"),  # british-ok: a file system type
    ("lustre", "rw,noflock", "rw", "delete", "unreliable"),  # british-ok: a file system type
    ("lustre", "rw", "rw", "delete", "unreliable"),  # british-ok: a file system type
    ("nfs", "rw", "rw,vers=3,nolock", "delete", "unreliable"),
    ("nfs", "rw", "rw,vers=3,local_lock=all", "delete", "unreliable"),
    ("nfs", "rw", "rw,vers=3,local_lock=posix", "delete", "unreliable"),
    ("cifs", "rw", "rw,nobrl", "delete", "unreliable"),
    ("beegfs", "rw", "rw", "delete", "unreliable"),
    ("fuse.sshfs", "rw", "rw", "delete", "unreliable"),
    ("fuseblk", "rw", "rw", "delete", "unreliable"),
    ("fuse", "rw", "rw", "delete", "unreliable"),        # a plain FUSE mount (an automounter, e.g. a research share)
    ("nfs", "rw", "rw,vers=3,local_lock=flock", "delete", "reliable"),
]


@pytest.mark.parametrize("fstype,mopts,sopts,mode,support", CASES)
def test_policy_from_mountinfo(monkeypatch, fstype, mopts, sopts, mode, support):
    monkeypatch.delenv("MACE_DB_JOURNAL", raising=False)
    monkeypatch.delenv("MACE_DB_LOCKING", raising=False)
    mi = _mi(fstype, mopts, sopts)
    m = D.mount_of("/data/proj/materials.db", mi)
    assert m.fstype == fstype and m.mount_point == "/data"
    assert D.lock_support(m) == support
    assert D.journal_policy("/data/proj/materials.db", mi) == (mode, support == "unreliable")
    # a path outside the mount is on the root file system: local
    assert D.journal_policy("/elsewhere/materials.db", mi) == ("wal", False)


def test_overrides(monkeypatch):
    mi = _mi("beegfs")
    monkeypatch.setenv("MACE_DB_LOCKING", "trust")
    assert D.lock_support(D.mount_of("/data/x.db", mi)) == "reliable"
    assert D.journal_policy("/data/x.db", mi) == ("delete", False)
    monkeypatch.setenv("MACE_DB_LOCKING", "degraded")
    assert D.journal_policy("/elsewhere/x.db", mi) == ("delete", True)
    monkeypatch.delenv("MACE_DB_LOCKING")
    monkeypatch.setenv("MACE_DB_JOURNAL", "wal")
    assert D.journal_policy("/data/x.db", _mi("gpfs")) == ("wal", False)
    monkeypatch.setenv("MACE_DB_JOURNAL", "delete")
    assert D.journal_policy("/elsewhere/x.db", mi)[0] == "delete"
    monkeypatch.delenv("MACE_DB_JOURNAL")
    assert D.mount_of("/x", "garbage without fields") is None
    assert D.journal_policy("/x/y.db", "") == ("wal", False)          # no mount table: local


# ------------------------------------------------------------------ C14-b
def _wal_db(path):
    c = sqlite3.connect(str(path))
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    c.execute("INSERT INTO t (v) VALUES ('a')")
    c.commit()
    c.close()


def test_switch_to_delete_and_held_wal(tmp_path, monkeypatch, capfd):
    db = tmp_path / "materials.db"
    _wal_db(db)
    monkeypatch.setenv("MACE_DB_JOURNAL", "delete")
    c = D.connect_rw(db)
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    c.close()
    # back to WAL, then a second process holds a read transaction open
    c = sqlite3.connect(str(db))
    c.execute("PRAGMA journal_mode=WAL")
    c.close()
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import sqlite3, sys, time
        c = sqlite3.connect({str(db)!r}, isolation_level=None)
        c.execute('BEGIN'); c.execute('SELECT count(*) FROM t').fetchone()
        print('held', flush=True); time.sleep(8)
        """)], stdout=subprocess.PIPE, text=True, env=_env())
    try:
        assert holder.stdout.readline().strip() == "held"
        D._said.clear()
        c = D.connect_rw(db)
        assert D.configure(c, db) == "wal"
        c.execute("INSERT INTO t (v) VALUES ('b')")
        c.commit()
        c.close()
        err = capfd.readouterr().err
        assert err.count("holds it in WAL mode") == 1
    finally:
        holder.kill()
        holder.wait()
    c = sqlite3.connect(str(db))
    assert c.execute("SELECT count(*) FROM t").fetchone()[0] == 2


# ------------------------------------------------------------------ C14-c
class _Busy(sqlite3.OperationalError):
    sqlite_errorcode = 5


class _BusySnapshot(sqlite3.OperationalError):
    sqlite_errorcode = 517            # SQLITE_BUSY_SNAPSHOT: an extended busy code


class _Other(sqlite3.OperationalError):
    sqlite_errorcode = 1


def test_retry_locked():
    clock = [0.0]
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise (_Busy if len(calls) == 1 else _BusySnapshot)("database is locked")
        return "ok"

    assert D.retry_locked(fn, sleep=lambda s: clock.__setitem__(0, clock[0] + s), clock=lambda: clock[0]) == "ok"
    assert len(calls) == 3

    def always():
        raise _Busy("database is locked")

    clock[0] = 0.0
    with pytest.raises(sqlite3.OperationalError, match="gave up after"):
        D.retry_locked(always, budget_s=5, sleep=lambda s: clock.__setitem__(0, clock[0] + s), clock=lambda: clock[0])
    n = []

    def other():
        n.append(1)
        raise _Other("no such table: x")

    with pytest.raises(_Other):
        D.retry_locked(other)
    assert len(n) == 1


# ------------------------------------------------------------------ C14-d
WRITER = textwrap.dedent("""
    import sqlite3, sys, time
    db, n = sys.argv[1], int(sys.argv[2])
    c = sqlite3.connect(db, timeout=30)
    c.execute('PRAGMA busy_timeout=30000')
    for i in range(1, n + 1):
        c.execute('INSERT INTO t (id, v) VALUES (?, ?)', (i, 'x' * 200))
        c.commit()
        if i == 1:
            print('started', flush=True)
        time.sleep(0.002)
    print('done', flush=True)
    """)


def test_backup_while_a_writer_commits(tmp_path):
    db = tmp_path / "materials.db"
    c = sqlite3.connect(str(db))
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    c.commit()
    c.close()
    w = subprocess.Popen(["nice", "-n", "10", sys.executable, "-c", WRITER, str(db), "3000"], stdout=subprocess.PIPE,
                         text=True, env=_env())
    try:
        assert w.stdout.readline().strip() == "started"
        copies = [D.snapshot_backup(db, tmp_path / "backups") for _ in range(3)]
    finally:
        w.wait(timeout=120)
    assert len({p.name for p in copies}) == 3 and all(p.is_file() for p in copies)   # same second: _2, _3
    for p in copies:
        b = sqlite3.connect(str(p))
        assert b.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        ids = [r[0] for r in b.execute("SELECT id FROM t ORDER BY id")]
        assert ids == list(range(1, len(ids) + 1))                  # a committed prefix
        b.close()


def test_prune_backups(tmp_path):
    d = tmp_path / "backups"
    d.mkdir()
    names = ["materials_backup_20260101_000000.db", "materials_backup_20260102_000000.db",
             "materials_backup_20260102_000000_2.db", "materials_backup_20260103_000000.db",
             "structures_backup_20260101_000000.db", "notes.txt", "materials_backup_x.db"]
    for n in names:
        (d / n).write_text("x")
    pruned = D.prune_backups(d, keep=3, out=None)
    assert pruned == ["materials_backup_20260101_000000.db"]
    assert sorted(p.name for p in d.iterdir()) == sorted(set(names) - set(pruned))


# ------------------------------------------------------------------ C14-e (golden)
def test_local_disk_stays_wal(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    shutil.copyfile(REPO / "examples" / "gui" / "outputs" / "Si_sp.out", root / "Si_sp.out")
    shutil.copyfile(REPO / "examples" / "gui" / "outputs" / "Si_sp.d12", root / "Si_sp.d12")
    db = tmp_path / "materials.db"
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "analyze", "--extract-properties",
                        str(root), "--db-path", str(db)], capture_output=True, text=True, env=_env(), cwd=tmp_path,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert db.is_file()
    c = sqlite3.connect(str(db))
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    c.close()


# ------------------------------------------------------------------ C14-f (large DB, stepped backup)
def test_large_db_stepped_backup_never_blocks_a_writer(tmp_path):
    if shutil.disk_usage(tmp_path).free < 2 * 1024 ** 3:
        pytest.skip("needs 2 GB free in the temp dir")
    db = tmp_path / "big.db"
    c = sqlite3.connect(str(db))
    c.execute("PRAGMA journal_mode=DELETE")
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v BLOB)")
    blob = os.urandom(64 * 1024)
    for i in range(300 * 16):           # ~300 MB
        c.execute("INSERT INTO t (v) VALUES (?)", (blob,))
    c.commit()
    c.close()
    writer = textwrap.dedent("""
        import sqlite3, sys, time
        c = sqlite3.connect(sys.argv[1], timeout=2)
        c.execute('PRAGMA busy_timeout=2000')
        t_end = time.time() + float(sys.argv[2]); n = 0
        while time.time() < t_end:
            c.execute("INSERT INTO t (v) VALUES (x'00')"); c.commit(); n += 1
            time.sleep(0.05)
        print('commits', n, flush=True)
        """)
    w = subprocess.Popen(["nice", "-n", "10", sys.executable, "-c", writer, str(db), "12"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env=_env())
    t0 = time.time()
    try:
        try:
            p = D.snapshot_backup(db, tmp_path / "backups", deadline_s=6, step_bytes=8 * 1024 * 1024)
            b = sqlite3.connect(str(p))
            assert b.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            b.close()
            outcome = "finished"
        except D.BackupTimeout as exc:
            assert "did not finish" in str(exc)
            assert not list((tmp_path / "backups").glob("*.db"))          # no partial copy left
            outcome = "timeout"
    finally:
        out, err = w.communicate(timeout=60)
    assert w.returncode == 0 and "database is locked" not in err, err
    print(f"stepped backup of 300 MB: {outcome} in {time.time() - t0:.1f} s; writer {out.strip()}")
    db.unlink()
    shutil.rmtree(tmp_path / "backups", ignore_errors=True)


# ------------------------------------------------------------------ C14-g (degraded writer lock)
DEGRADED_WRITER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[3])
    from mace.database import dbsafety as D
    log = open(sys.argv[2], 'a')
    for i in range(5):
        with D.write_guard(sys.argv[1]):
            t0 = time.time(); time.sleep(0.05); t1 = time.time()
            log.write(f'{t0} {t1}\\n'); log.flush()
        time.sleep(0.01)
    """)


def test_degraded_writers_serialize(tmp_path):
    db = tmp_path / "materials.db"
    db.write_bytes(b"")
    log = tmp_path / "timeline.txt"
    env = _env(MACE_DB_LOCKING="degraded")
    ps = [subprocess.Popen([sys.executable, "-c", DEGRADED_WRITER, str(db), str(log), str(REPO)], env=env,
                           stderr=subprocess.PIPE, text=True) for _ in range(2)]
    for p in ps:
        assert p.wait(timeout=60) == 0, p.stderr.read()
    spans = sorted(tuple(map(float, ln.split())) for ln in log.read_text().splitlines())
    assert len(spans) == 10
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert b0 >= a1, (a0, a1, b0, b1)                  # no overlap
    assert not Path(str(db) + ".mace-writer.lock").exists()
    # a stale lock (older than 120 s) is taken over
    lock = Path(str(db) + ".mace-writer.lock")
    lock.write_text("crashed\n")
    old = time.time() - 300
    os.utime(lock, (old, old))
    with D.writer_lock(db, wait_s=5):
        assert lock.read_text() != "crashed\n"
    assert not lock.exists()


@pytest.mark.parametrize("folder", ["run#2", "what?now", "100%done"])
def test_read_only_opens_survive_uri_characters_in_the_path(tmp_path, folder):
    """A '#', '?' or '%' in a folder name must not end the path of the
    read-only URI: the backup would copy a new empty file instead (and
    leave it behind), and the build preview would see no database."""
    from mace.utils import db_build
    d = tmp_path / folder
    d.mkdir()
    db = d / "materials.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE calculations (calc_id TEXT)")
    c.execute("INSERT INTO calculations VALUES ('c1')")
    c.commit()
    c.close()
    before = sorted(p.name for p in tmp_path.rglob("*"))
    dst = D.snapshot_backup(db, d / "backups")
    b = sqlite3.connect(dst)
    assert b.execute("SELECT calc_id FROM calculations").fetchall() == [("c1",)]
    b.close()
    conn = db_build.open_ro(db)
    assert conn is not None
    conn.close()
    after = sorted(p.name for p in tmp_path.rglob("*") if "backups" not in p.parts)
    assert after == before          # no stray file created by a truncated URI
