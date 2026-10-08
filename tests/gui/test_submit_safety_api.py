"""Submit with the file-safety pieces of Area C (PLAN_submit D37, D39): File
History before the CLI replaces a file, a database backup before every
tracked submit (the newest three kept), finished outputs copied (never
moved) before an overwrite, and one GUI writer per database.

These need mace/gui/history.py, mace/database/dbsafety.py and
mace/gui/dbwriters.py (Area C); without them the tests skip."""
import json
import os
import sqlite3
import stat
import threading
import time
from pathlib import Path

import pytest

import submit_util as SU
from conftest import TEST_DATA

pytest.importorskip("mace.gui.history")
pytest.importorskip("mace.gui.dbwriters")
dbsafety = pytest.importorskip("mace.database.dbsafety")
if not hasattr(dbsafety, "snapshot_backup"):
    pytest.skip("dbsafety.snapshot_backup is not here", allow_module_level=True)


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("safety")
    hist = tmp / "history"
    s = SU.Served(tmp, MACE_GUI_HISTORY_DIR=str(hist))
    s.hist = hist
    yield s
    s.stop()


def _entries(hist, rel):
    out = []
    for idx in hist.rglob("index.jsonl"):
        for line in idx.read_text().splitlines():
            r = json.loads(line)
            if r.get("rel_path") == rel:
                out.append((idx.parent, r))
    return out


def test_history_before_write(srv):
    d = srv.decks("h/decks", ["a"])
    sh = d / "a.sh"
    sh.write_text("#!/bin/bash\n# an older job script\n")
    old = sh.read_bytes()
    st, prev = srv.preview(["h/decks/a.d12"], options={"overwrite_sh": True})
    assert "h/decks/a.sh" in prev["modified"]
    st, res = srv.run(prev["preview_id"])
    assert res["code"] == 0, res
    (fdir, e), = [x for x in _entries(srv.hist, "h/decks/a.sh") if not x[1].get("created")]
    assert (fdir / e["id"]).read_bytes() == old and sh.read_bytes() != old


def test_history_refusal(tmp_path):
    hist = tmp_path / "ro"
    hist.mkdir()
    s = SU.Served(tmp_path, MACE_GUI_HISTORY_DIR=str(hist))
    try:
        d = s.decks("h/decks", ["a"])
        (d / "a.sh").write_text("#!/bin/bash\n")
        os.chmod(hist, stat.S_IRUSR | stat.S_IXUSR)
        st, prev = s.preview(["h/decks/a.d12"], options={"overwrite_sh": True})
        st, res = s.run(prev["preview_id"])
        assert st != 200 or res.get("code") != 0
        assert not s.calls("sbatch")
    finally:
        os.chmod(hist, stat.S_IRWXU)
        s.stop()


def test_db_backup_every_tracked_submit(srv):
    d = srv.decks("bk/decks", ["a", "b"])
    for k in range(5):
        st, prev = srv.preview([f"bk/decks/{'ab'[k % 2]}.d12"], options={"track": True})
        st, res = srv.run(prev["preview_id"], confirm_duplicate=True, confirm_overwrite=True)
        assert res["code"] == 0, res
        if k:
            assert res["backups"], res
        time.sleep(1.1)
    backups = sorted((srv.root / "bk" / "decks" / "backups").glob("materials_backup_*.db"))
    assert len(backups) == 3                      # the newest three are kept
    for b in backups:
        with sqlite3.connect(b) as con:
            assert con.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_overwrite_copies_not_moves(srv):
    d = srv.decks("ow/decks", ["m"])
    src = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if src.is_file():
        (d / "m.out").write_bytes(src.read_bytes())
        (d / "m.f9").write_bytes((TEST_DATA / "OPT" / "1_dia_opt_rev1.f9").read_bytes())
    else:
        (d / "m.out").write_text("TOTAL CPU TIME =    1.0\n")
        (d / "m.f9").write_bytes(b"\0" * 1000)
    out_bytes, f9_bytes = (d / "m.out").read_bytes(), (d / "m.f9").read_bytes()
    st, prev = srv.preview(["ow/decks/m.d12"])
    st, res = srv.run(prev["preview_id"], confirm_overwrite=True)
    assert res["code"] == 0
    (fdir, e), = _entries(srv.hist, "ow/decks/m.out")
    assert e["kind"] == "output" and (fdir / e["id"]).read_bytes() == out_bytes
    (fdir, e), = _entries(srv.hist, "ow/decks/m.f9")
    assert (fdir / e["id"]).read_bytes() == f9_bytes
    assert (d / "m.f9").read_bytes() == f9_bytes            # copied, never moved


def test_db_writer_lock(tmp_path):
    s = SU.Served(tmp_path, {"sbatch_delay": 4})
    try:
        d = s.decks("wl/decks", ["a", "b"])
        st, prev = s.preview(["wl/decks/a.d12"], options={"track": True})
        st, res = s.run(prev["preview_id"])                  # creates wl/decks/materials.db
        st, prev = s.preview(["wl/decks/b.d12"], options={"track": True})
        r = s.req("api/submit/run", {"preview_id": prev["preview_id"]})
        assert r[0] in (200, 202)
        time.sleep(1.5)
        st, _, info = s.req("api/info")
        code, _, body = s.req("api/db/build", {"folder": "wl/decks", "db": "wl/decks/materials.db", "confirm": True,
                                               "root_gen": info.get("root_gen")})
        assert code == 409 and "Submit (mace submit --track) is writing this database" in body["error"], body
        s.task_result(r)
        # an untracked submit takes no lock
        st, prev = s.preview(["wl/decks/a.d12"])
        r = s.req("api/submit/run", {"preview_id": prev["preview_id"], "confirm_duplicate": True})
        time.sleep(1.0)
        from mace.gui import dbwriters  # noqa: F401
        s.task_result(r)
    finally:
        s.stop()
