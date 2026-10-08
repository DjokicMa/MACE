"""Simultaneous writers on one materials.db, and writers killed mid-transaction
(PLAN_analysis C16, Marcus's "simultaneous-writer test").

Writers, each a real subprocess at nice 10 with one thread:
* W-callback: the real job callback (manager.py --callback-mode completion,
  as the job script's last line runs it) once per tracked job, over 20
  copied completed outputs, with FakeSlurm (no job is in the queue);
* W-build: the real `mace_cli analyze --extract-properties DIR --register`
  on a second copied folder, 3 times;
* W-hammer: MaterialDatabase (the callback's own API), 400 transactions of
  one material and its property on one connection, 0-20 ms apart.
Reader: the real `mace_cli gui --dir` server, paging the Database view's
routes every 200 ms.

Cases: MACE_DB_JOURNAL in {wal, delete} x {W-callback + W-hammer, W-build +
W-hammer}, plus delete with MACE_DB_LOCKING=degraded and W-callback +
W-hammer; then W-hammer SIGKILLed at 10 random points in delete and in wal
mode. One case at a time across xdist workers (a lock file), at most three
child processes per case.
"""
import fcntl
import json
import os
import random
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

sys.path.insert(0, str(Path(__file__).resolve().parent / "gui"))
import guiharness as H  # noqa: E402

pytestmark = [pytest.mark.dbconcurrency, pytest.mark.xdist_group("dbconcurrency")]

SP_OUT = "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out"
SP_D12 = "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12"
MANAGER = REPO_ROOT / "mace" / "queue" / "manager.py"
N_CALLBACK = 20
N_HAMMER = 400


@pytest.fixture(scope="module", autouse=True)
def one_case_at_a_time(tmp_path_factory):
    """xdist workers share basetemp's parent: one module run of this file at a time."""
    lock = tmp_path_factory.getbasetemp().parent / "mace_dbconcurrency.lock"
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield
        fcntl.flock(fh, fcntl.LOCK_UN)


def _need_corpus():
    if not (TEST_DATA / SP_OUT).is_file():
        pytest.skip("test/ data corpus not present (gitignored)")


def _env(journal, locking=None, extra=None):
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e.update(OMP_NUM_THREADS="1", MACE_DB_JOURNAL=journal)
    if locking:
        e["MACE_DB_LOCKING"] = locking
    e.update(extra or {})
    return e


def _nice(argv):
    return ["nice", "-n", "10", *argv] if shutil.which("nice") else argv


HAMMER = r'''
import random, sys, time
from datetime import datetime
sys.path.insert(0, sys.argv[1])
from mace.database.materials import MaterialDatabase
db = MaterialDatabase(sys.argv[2], ase_db_path=sys.argv[2] + ".ase.db")
n, prefix = int(sys.argv[3]), sys.argv[4]
start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
for k in range(start, n):
    now = datetime.now().isoformat()
    with db._get_connection() as c:
        c.execute("INSERT INTO materials (material_id, formula, created_at, updated_at) VALUES (?, ?, ?, ?)",
                  (f"{prefix}{k:04d}", "C2", now, now))
        c.execute("INSERT INTO properties (material_id, property_category, property_name, property_value, extracted_at, "
                  "extractor_script) VALUES (?, 'test', 'hammer', ?, ?, 'hammer')", (f"{prefix}{k:04d}", float(k), now))
    print(k, flush=True)
    time.sleep(random.random() * 0.02)
'''


def _hammer(db, env, n=N_HAMMER, prefix="h", start=0):
    return subprocess.Popen(_nice([sys.executable, "-c", HAMMER, str(REPO_ROOT), str(db), str(n), prefix, str(start)]),
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _tracked_jobs(root, db, env):
    """20 tracked jobs (SUBMITTED rows with job ids 900001..) whose completed outputs are in place."""
    sys.path.insert(0, str(REPO_ROOT))
    code = r'''
import sys, shutil
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from mace.database.materials import MaterialDatabase
root, db, out, d12, n = Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
m = MaterialDatabase(db, ase_db_path=db + ".ase.db")
for k in range(n):
    name = f"job{k:02d}"
    wd = root / name
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(d12, wd / f"{name}.d12")
    shutil.copyfile(out, wd / f"{name}.out")
    m.create_material(name, "C2", 227)
    cid = m.create_calculation(material_id=name, calc_type="SP", input_file=str(wd / f"{name}.d12"), work_dir=str(wd))
    m.update_calculation_status(cid, "submitted", slurm_job_id=str(900001 + k))
'''
    p = subprocess.run([sys.executable, "-c", code, str(REPO_ROOT), str(root), str(db), str(TEST_DATA / SP_OUT),
                        str(TEST_DATA / SP_D12), str(N_CALLBACK)], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]


def _fakeslurm(tmp):
    """sbatch and squeue that do nothing (no job is in the queue); sbatch calls are logged."""
    b = tmp / "fakebin"
    b.mkdir(exist_ok=True)
    log = tmp / "sbatch.log"
    (b / "sbatch").write_text(f"#!/bin/sh\necho \"$@\" >> {log}\necho Submitted batch job 999999\n")
    for t in ("squeue", "sacct", "scontrol"):
        (b / t).write_text("#!/bin/sh\nexit 0\n")
    for f in b.iterdir():
        f.chmod(0o755)
    return b, log


class WCallback(threading.Thread):
    """The real completion callback, once per tracked job, in turn."""

    def __init__(self, root, db, env):
        super().__init__(daemon=True)
        self.root, self.db, self.env, self.codes, self.errors = root, db, env, [], []

    def run(self):
        for k in range(N_CALLBACK):
            name = f"job{k:02d}"
            env = dict(self.env, SLURM_JOB_ID=str(900001 + k), PYTHONPATH=str(REPO_ROOT), MACE_HOME=str(REPO_ROOT))
            p = subprocess.run(_nice([sys.executable, str(MANAGER), "--callback-mode", "completion", "--db-path", str(self.db)]),
                               cwd=str(self.root / name), env=env, capture_output=True, text=True, timeout=600)
            self.codes.append(p.returncode)
            if p.returncode or "database is locked" in (p.stderr + p.stdout).replace("retry", ""):
                self.errors.append((name, p.returncode, (p.stdout + p.stderr)[-1500:]))


class WBuild(threading.Thread):
    """The real database build from a folder, three times."""

    def __init__(self, folder, db, env):
        super().__init__(daemon=True)
        self.folder, self.db, self.env, self.codes, self.errors = folder, db, env, [], []

    def run(self):
        for _ in range(3):
            p = subprocess.run(_nice([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--extract-properties",
                                      str(self.folder), "--db-path", str(self.db), "--register"]),
                               cwd=str(self.folder.parent), env=self.env, capture_output=True, text=True, timeout=900)
            self.codes.append(p.returncode)
            if p.returncode:
                self.errors.append((p.returncode, (p.stdout + p.stderr)[-1500:]))


class Reader(threading.Thread):
    """The real `mace gui` server's Database routes, every 200 ms."""

    def __init__(self, base, rel):
        super().__init__(daemon=True)
        self.base, self.rel, self.stop = base, rel, threading.Event()
        self.statuses, self.counts, self.bad = [], [], []

    def get(self, route, **q):
        q["path"] = self.rel
        req = urllib.request.Request(self.base + f"api/{route}?" + urllib.parse.urlencode(q),
                                     headers={"X-Mace-Token": H.TOKEN})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    def run(self):
        while not self.stop.is_set():
            for route in ("db/stats", "db/page"):
                st, body = self.get(route, limit=50)
                self.statuses.append(st)
                if st == 200 and route == "db/stats":
                    self.counts.append(body["n_materials"])
                elif st == 503 and not (body or {}).get("busy"):
                    self.bad.append((st, body))
                elif st not in (200, 503):
                    self.bad.append((st, body))
            time.sleep(0.2)


def _server(root, xdg, journal, locking=None):
    """The real `mace gui` (the GUI tests' harness), with the same journal settings as the writers."""
    extra = {"MACE_GUI_NO_WARMUP": "1", "MACE_GUI_TEST_HOOKS": "1", "MACE_DB_JOURNAL": journal, "OMP_NUM_THREADS": "1"}
    if locking:
        extra["MACE_DB_LOCKING"] = locking
    proc, base, _ = H.start_server(root, xdg, **extra)
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
    return proc, base


def _stop(p):
    p.terminate()
    try:
        p.wait(15)
    except subprocess.TimeoutExpired:
        p.kill()


def _q(db, sql, *a):
    c = sqlite3.connect(str(db), timeout=30)
    try:
        return c.execute(sql, a).fetchall()
    finally:
        c.close()


def _check_hammer(db, prefix="h", n=N_HAMMER):
    mats = _q(db, "SELECT count(*) FROM materials WHERE material_id LIKE ?", prefix + "%")[0][0]
    props = _q(db, "SELECT count(*) FROM properties WHERE property_name = 'hammer' AND material_id LIKE ?", prefix + "%")[0][0]
    lone = _q(db, "SELECT count(*) FROM materials m WHERE material_id LIKE ? AND "
                  "(SELECT count(*) FROM properties p WHERE p.material_id = m.material_id AND p.property_name = 'hammer') != 1",
              prefix + "%")[0][0]
    return mats, props, lone


CASES = [("wal", None, "callback"), ("wal", None, "build"), ("delete", None, "callback"), ("delete", None, "build"),
         ("delete", "degraded", "callback")]


@pytest.mark.parametrize("journal,locking,pair", CASES, ids=[f"{j}-{l or 'trust'}-{p}" for j, l, p in CASES])
def test_simultaneous_writers(tmp_path, journal, locking, pair, record_property):
    _need_corpus()
    t0 = time.monotonic()
    root = tmp_path / "root"
    root.mkdir()
    db = root / "materials.db"
    fakebin, sblog = _fakeslurm(tmp_path)
    env = _env(journal, locking, {"PATH": f"{fakebin}{os.pathsep}{os.environ.get('PATH', '')}", "HOME": str(tmp_path / "home"),
                                   "USER": "tester"})
    (tmp_path / "home").mkdir()
    if pair == "callback":
        _tracked_jobs(root / "jobs", str(db), env)
    else:
        folder = root / "runs"
        folder.mkdir()
        for r in ("OPT/1_dia_opt_rev1.out", "OPT/1_dia_opt_rev1.d12", SP_OUT, SP_D12,
                  "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out", "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.d3"):
            shutil.copyfile(TEST_DATA / r, folder / Path(r).name)
        _q_init = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]);"
                                  "from mace.database.materials import MaterialDatabase; MaterialDatabase(sys.argv[2], ase_db_path=sys.argv[2] + '.ase.db')",
                                  str(REPO_ROOT), str(db)], env=env, capture_output=True, text=True, timeout=120)
        assert _q_init.returncode == 0, _q_init.stderr
    server, base = _server(root, tmp_path / "xdg", journal, locking)
    reader = Reader(base, "materials.db")
    try:
        reader.start()
        hammer = _hammer(db, env)
        other = WCallback(root / "jobs", db, env) if pair == "callback" else WBuild(root / "runs", db, env)
        other.start()
        out, err = hammer.communicate(timeout=900)
        other.join(timeout=1800)
        time.sleep(0.6)
    finally:
        reader.stop.set()
        reader.join(timeout=60)
        _stop(server)
    assert hammer.returncode == 0, err[-3000:]
    assert "database is locked" not in err, err[-3000:]
    assert not other.errors, other.errors
    assert _q(db, "PRAGMA integrity_check")[0][0] == "ok"
    assert _check_hammer(db) == (N_HAMMER, N_HAMMER, 0)
    if pair == "callback":
        st = _q(db, "SELECT status, count(*) FROM calculations GROUP BY status")
        assert dict(st) == {"completed": N_CALLBACK}, st
        assert not sblog.exists() or not sblog.read_text().strip()
    assert reader.statuses and not reader.bad, reader.bad[:5]
    assert reader.counts == sorted(reader.counts), "the page count went down"
    mode = _q(db, "PRAGMA journal_mode")[0][0]
    assert mode == journal, mode
    record_property("seconds", round(time.monotonic() - t0, 1))
    record_property("reader_503", reader.statuses.count(503))


@pytest.mark.parametrize("journal", ["delete", "wal"])
def test_killed_writer_leaves_a_whole_database(tmp_path, journal, record_property):
    """SIGKILL the W-hammer writer at 10 random points: the database stays whole, every material keeps exactly
    its one property, the GUI never reads it immutable (busy 'interrupted' while a hot journal is there), and
    the next writer repairs it without help."""
    _need_corpus()
    root = tmp_path / "root"
    root.mkdir()
    db = root / "materials.db"
    env = _env(journal)
    p = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]);"
                        "from mace.database.materials import MaterialDatabase; MaterialDatabase(sys.argv[2], ase_db_path=sys.argv[2] + '.ase.db')",
                        str(REPO_ROOT), str(db)], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    server, base = _server(root, tmp_path / "xdg", journal)
    reader = Reader(base, "materials.db")
    rnd = random.Random(16)
    interrupted = 0
    try:
        done = 0
        for kill in range(10):
            h = _hammer(db, env, n=done + 60, prefix="h", start=done)
            time.sleep(0.2 + rnd.random() * 0.6)
            h.send_signal(signal.SIGKILL)
            h.wait(30)
            hot = Path(str(db) + "-journal").exists() and Path(str(db) + "-journal").stat().st_size > 0
            st, body = reader.get("db/stats")
            assert st in (200, 503), (st, body)
            if st == 503:
                assert body["busy"] and body["reason"] in ("interrupted", "busy"), body
                interrupted += body["reason"] == "interrupted"
            elif hot and journal == "delete":
                # SQLite rolled the journal back for the reader (it could write the folder): fine, never immutable
                pass
            assert _q(db, "PRAGMA integrity_check")[0][0] == "ok"
            mats, props, lone = _check_hammer(db)
            assert mats == props and lone == 0, (mats, props, lone)
            done = mats
        # the next writer reopens it without repair; then the reader answers 200
        h = _hammer(db, env, n=done + 5, prefix="h", start=done)
        assert h.wait(120) == 0
        st, body = reader.get("db/stats")
        assert st == 200 and body["n_materials"] == done + 5
        with urllib.request.urlopen(urllib.request.Request(base + "api/test/dblog", headers={"X-Mace-Token": H.TOKEN})) as r:
            assert json.loads(r.read())["immutable_opens"] == []          # never an immutable read
    finally:
        _stop(server)
    record_property("interrupted_answers", interrupted)
