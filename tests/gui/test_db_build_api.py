"""Create or Update Database from a Folder (PLAN_analysis C10) on the real
`mace gui` subprocess: the dialog facts, the preview (the CLI's own
`--dry-run --json`), the build (the CLI with `--progress-json`) as a
cancelable task holding the one GUI writer lock of that database, the
Database view's "building" answer, and the refusals. Real outputs copied
from the corpus; nothing of it is committed."""
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import runcorpus as RC

# the repository only: mace/ itself on sys.path would make `import queue` (urllib3, selenium) MACE's
# queue package for every test file that runs after this one
sys.path[:0] = [p for p in (str(H.REPO_ROOT),) if p not in sys.path]

MIXED = ["OPT/1_dia_opt_rev1.out", "OPT/1_dia_opt_rev1.d12",
         "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out", "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12",
         "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.out", "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.d3",
         "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out", "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.d3",
         "FAILED_QA/3_dia3_opt_killed_signal9_oom.out"]


def _mixed(dst: Path) -> Path:
    for r in MIXED:
        RC.copy_into(dst, RC.CORPUS / r)
    return dst


def _env():
    e = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _cli(cwd, *args):
    p = subprocess.run([sys.executable, str(H.REPO_ROOT / "mace_cli"), "--no-banner", *args], capture_output=True,
                       text=True, env=_env(), cwd=str(cwd), timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return p


def _q(db, sql):
    c = sqlite3.connect(str(db))
    try:
        return c.execute(sql).fetchall()
    finally:
        c.close()


def _post(base, route, body):
    return U.req(base, "api/" + route, body)


def _get(base, route, **q):
    return U.req(base, f"api/{route}?" + urllib.parse.urlencode(q))


def _wait(base, r, timeout=600):
    """The finished task's record and result (None unless done)."""
    st, _, body = r
    if st == 200:
        return body["task"], body
    assert st == 202, body
    tid = body["task"]["id"]
    end = time.time() + timeout
    while time.time() < end:
        s, _, lst = U.req(base, "api/tasks")
        rec = next((t for t in (lst.get("tasks") if isinstance(lst, dict) else lst) if t["id"] == tid), None)
        if rec and rec["state"] in ("done", "failed", "canceled"):
            if rec["state"] == "done":
                s2, _, res = U.req(base, f"api/tasks/{tid}/result")
                return rec, res
            return rec, None
        time.sleep(0.2)
    raise AssertionError("task did not finish")


def _md5_tree(d: Path, skip=()):
    out = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and not any(s in p.parts or p.name.startswith(s) for s in skip):
            out[p.relative_to(d).as_posix()] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    RC.need(RC.DIA_OPT)
    tmp = tmp_path_factory.mktemp("dbbuild")
    root = tmp / "root"
    _mixed(root / "runs")
    _mixed(root / "runs2")
    outside = tmp / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield base, root, tmp
    proc.terminate()
    proc.wait(timeout=15)


def test_info_names_the_database_and_the_command(served):
    base, root, _ = served
    st, _, r = _get(base, "db/build/info", folder="runs")
    assert st == 200, r
    assert r["db"] == "runs/materials.db" and r["source"] == "new" and r["exists"] is False and r["inside"]
    assert r["command"] == "mace analyze --extract-properties runs --db-path runs/materials.db --register --skip-unchanged"
    assert r["outputs"] == 5 and r["guarantee"].startswith("Nothing is deleted from the database.")
    # the Backup tick of a new database is in the command the build runs (db_build_route's argv)
    st, _, r = _get(base, "db/build/info", folder="runs", backup="1", keep="4")
    assert r["command"] == ("mace analyze --extract-properties runs --db-path runs/materials.db --register "
                            "--skip-unchanged --backup --keep-backups 4")
    from mace.gui import dbbuild
    opts = dbbuild.options({"backup": True, "keep": 4}, False)
    assert r["command"] == dbbuild.command_text(dbbuild.cli_args("runs", "runs/materials.db", opts["register"],
                                                                 opts["skip_unchanged"], opts["backup"], opts["keep"]))


def test_preview_equals_the_cli_dry_run_and_writes_nothing(served):
    base, root, _ = served
    before = _md5_tree(root / "runs")
    rec, res = _wait(base, _post(base, "db/build/preview", {"folder": "runs", "register": True}))
    assert rec["state"] == "done", rec
    cli = json.loads(_cli(root, "analyze", "--extract-properties", "runs", "--db-path", "runs/materials.db",
                          "--register", "--dry-run", "--json").stdout.strip().splitlines()[-1])
    assert res["counts"] == cli["counts"]
    assert res["counts"]["new"] >= 4 and res["counts"]["unreadable"] == 0
    assert sorted(f["rel"] for f in res["files"]) == sorted("runs/" + Path(f["file"]).name for f in cli["files"])
    assert _md5_tree(root / "runs") == before and not (root / "runs" / "materials.db").exists()


def test_build_creates_then_updates_with_a_backup(served):
    base, root, _ = served
    db = root / "runs" / "materials.db"
    U.req(base, "api/test/gate/reset", {})
    rec, res = _wait(base, _post(base, "db/build", {"folder": "runs", "register": True, "skip_unchanged": True,
                                                     "confirm": True}))
    assert rec["state"] == "done", rec
    # the argv the server ran is the shown command plus --progress-json
    st, _, log = U.req(base, "api/test/log")
    argv = [x["db_build_argv"] for x in (log.get("log") if isinstance(log, dict) else log) if "db_build_argv" in x][-1]
    assert "mace " + " ".join(argv[:-1]) == res["command"] and argv[-1] == "--progress-json"
    mats = _q(db, "SELECT material_id, formula FROM materials")
    assert mats and all(f != "Unknown" for _, f in mats)
    assert not (root / "runs" / "backups").exists()            # nothing to back up on create
    n_props = _q(db, "SELECT count(*) FROM properties")[0][0]
    # change one output's content: the preview says 1 changed, the update backs up first
    sp = root / "runs" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out"
    with open(sp, "a") as fh:
        fh.write("\n")
    _, prev = _wait(base, _post(base, "db/build/preview", {"folder": "runs", "register": True}))
    assert prev["counts"]["changed"] == 1, prev["counts"]
    st, _, info = _get(base, "db/build/info", folder="runs")
    assert info["exists"] and info["source"] == "folder" and "--backup --keep-backups 3" in info["command"]
    rec, res = _wait(base, _post(base, "db/build", {"folder": "runs", "confirm": True, "backup": False}))
    assert rec["state"] == "done"
    assert res["backup"] and res["backup"].startswith("runs/backups/materials_backup_")    # locked on: a backup anyway
    assert len(list((root / "runs" / "backups").glob("materials_backup_*.db"))) == 1
    assert _q(db, "SELECT count(*) FROM properties")[0][0] >= n_props
    ex = [f["class"] for f in res["files"] if f["status"] == "extracted"]
    assert ex.count("changed") == 1 and "unchanged" not in ex and res["changed"] == 1     # incomplete runs are always re-read


def test_gui_and_cli_builds_are_identical(served):
    """C10-f: the GUI's database and the shown command's database (second copy)."""
    base, root, tmp = served
    rec, res = _wait(base, _post(base, "db/build", {"folder": "runs2", "db": "runs2/gui.db", "confirm": True}))
    assert rec["state"] == "done"
    cli_dir = tmp / "clicopy"
    _mixed(cli_dir / "runs2")
    cmd = res["command"].split()[1:]
    cmd[cmd.index("--db-path") + 1] = "runs2/gui.db"
    _cli(cli_dir, *cmd)

    def dump(db):
        c = sqlite3.connect(str(db))
        try:
            counts = {t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                      for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            mats = sorted(c.execute("SELECT material_id, formula, space_group, dimensionality FROM materials"))
            props = sorted(c.execute("SELECT material_id, property_name, property_value, property_unit FROM properties"))
            calcs = sorted(c.execute("SELECT material_id, calc_type, status FROM calculations"))
        finally:
            c.close()
        return counts, mats, props, calcs
    a, b = dump(root / "runs2" / "gui.db"), dump(cli_dir / "runs2" / "gui.db")
    assert a == b


def test_refusals_create_nothing(served):
    base, root, tmp = served
    body = {"folder": "runs", "confirm": True}
    assert _post(base, "db/build", {"folder": "runs"})[0] == 400                          # no confirm
    assert _post(base, "db/build", {**body, "db": "../x.db"})[0] == 403                   # outside
    assert _post(base, "db/build", {**body, "db": str(tmp / "outside" / "a.db")})[0] == 403
    assert _post(base, "db/build", {**body, "db": "escape/a.db"})[0] == 403               # symlink escape
    assert _post(base, "db/build", {**body, "folder": "../"})[0] == 403
    assert _post(base, "db/build", {**body, "db": "runs/notes.txt"})[0] == 400
    assert U.req(base, "api/db/build", body, token=False)[0] == 403
    assert U.req(base, "api/db/build", body, headers={"X-Requested-With": "x"})[0] == 403
    st, _, info = _get(base, "db/build/info", folder="runs", db=str(tmp / "outside" / "b.db"))
    assert st == 200 and info["inside"] is False and info["outside_note"].startswith("This database is outside")
    assert not list((tmp / "outside").iterdir())


@pytest.fixture(scope="module")
def slow(tmp_path_factory):
    """A server whose CLI children start 6 s late (test hook): a build holds its lock long enough to look at."""
    RC.need(RC.DIA_OPT)
    tmp = tmp_path_factory.mktemp("dbbuildslow")
    root = tmp / "root"
    d = root / "many"
    src = [RC.CORPUS / r for r in MIXED if r.endswith(".out") and "FAILED" not in r]
    d.mkdir(parents=True)
    for k in range(10):
        for p in src:
            shutil.copy2(p, d / f"c{k}_{p.name}")
    _cli(root, "analyze", "--extract-properties", "many", "--db-path", "many/materials.db", "--register")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1",
                                   MACE_GUI_TEST_CLI_DELAY="6")
    yield base, root
    proc.terminate()
    proc.wait(timeout=15)


def test_second_writer_and_database_view_during_a_build(slow):
    """C10-d, C10-e: during a build the Database view answers "building" (no
    read of the file, no immutable open) and a second GUI writer gets 409."""
    base, root = slow
    st, _, r = _post(base, "db/build", {"folder": "many", "confirm": True, "skip_unchanged": False, "wait": 0})
    assert st == 202
    tid = r["task"]["id"]
    st2, _, r2 = _post(base, "db/build", {"folder": "many", "confirm": True})
    assert st2 == 409 and "is writing this database" in r2["error"]
    st3, _, r3 = _get(base, "db/stats", path="many/materials.db")
    assert st3 == 503 and r3["reason"] == "building"
    st4, _, r4 = _get(base, "db/stamp", path="many/materials.db")
    assert st4 == 200 and r4["building"]
    _wait(base, (202, {}, {"task": {"id": tid}}), timeout=900)
    st5, _, r5 = _get(base, "db/stats", path="many/materials.db")
    assert st5 == 200 and r5["n_materials"] >= 1
    st6, _, r6 = _get(base, "db/stamp", path="many/materials.db")
    assert r6["building"] is None


def test_cancel_keeps_the_database_whole(slow):
    """C10-c: cancel mid-build: Canceled, integrity ok, the summary names the
    backup, and the folder changed only in the database files and backups/."""
    base, root = slow
    before = _md5_tree(root / "many", skip=("materials.db", "backups", "structures.db", ".materials.db.building"))
    st, _, r = _post(base, "db/build", {"folder": "many", "confirm": True, "skip_unchanged": False, "wait": 0})
    tid = r["task"]["id"]
    end = time.time() + 300
    while time.time() < end:
        _, _, lst = U.req(base, "api/tasks")
        rec = next(t for t in (lst.get("tasks") if isinstance(lst, dict) else lst) if t["id"] == tid)
        if rec["progress"] and rec["progress"] > 0.05:
            break
        time.sleep(0.2)
    U.req(base, f"api/tasks/{tid}/cancel", {})
    rec, _ = _wait(base, (202, {}, {"task": {"id": tid}}))
    assert rec["state"] == "canceled"
    assert "the backup is many/backups/materials_backup_" in rec["log"], rec["log"]
    assert _q(root / "many" / "materials.db", "PRAGMA integrity_check")[0][0] == "ok"
    assert _md5_tree(root / "many", skip=("materials.db", "backups", "structures.db", ".materials.db.building")) == before


def test_the_next_build_continues_a_stopped_one(slow):
    """1.1.6: the canceled build above left its marker; the next build with
    'Only new and changed files' skips what it committed and says so, in its
    progress and in its summary (the CLI's own message); then the marker goes."""
    base, root = slow
    marker = root / "many" / ".materials.db.building"
    assert marker.is_file()
    rec, res = _wait(base, _post(base, "db/build", {"folder": "many", "confirm": True, "skip_unchanged": True}), timeout=900)
    assert rec["state"] == "done", rec
    r = res["resumed"]
    assert r and r["total"] == 40 and 1 <= r["already"] <= 40, r
    assert res["message"].startswith(f"Resumed: {r['already']} of 40 already done. Database updated:"), res["message"]
    assert sum(1 for f in res["files"] if f.get("class") == "unchanged") == r["already"]
    assert not marker.exists()
    rec, res = _wait(base, _post(base, "db/build", {"folder": "many", "confirm": True, "skip_unchanged": True}), timeout=900)
    assert rec["state"] == "done" and res["resumed"] is None and not res["message"].startswith("Resumed")
