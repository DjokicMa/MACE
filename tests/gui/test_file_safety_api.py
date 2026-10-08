"""The GUI never writes over calculation outputs (PLAN_analysis C19) and
keeps the previous version of every file it replaces in File History
(C20). The real `mace gui` subprocess on a copy of examples/gui; protected
files are real copies (job scripts from tests/data/job_scripts, outputs from
examples/gui)."""
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

sys.path[:0] = [p for p in (str(H.REPO_ROOT),) if p not in sys.path]
JOB_SCRIPTS = H.REPO_ROOT / "tests" / "data" / "job_scripts"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("filesafety")
    root = H.examples_copy(tmp)
    prot = root / "prot"
    prot.mkdir()
    src_out = H.EXAMPLES / "outputs" / "Si_sp.out"
    for name in ("run.out", "slurm-123.out", "job.o4242", "job.e4242", "x.f9", "fort.9", "fort.98", "x.f98", "fort.20",
                 "x_band.BAND.DAT", "fort.25", "x.f25", "x_DENS.CUBE", "SCFOUT.LOG", "m.db", "m.db-wal", "m.db-journal",
                 "m.db.mace-writer.lock"):
        shutil.copyfile(src_out, prot / name)
    scripts = sorted(JOB_SCRIPTS.glob("*.sh"))
    assert scripts, "tests/data/job_scripts holds the v1.1.1 job script fixtures"
    for s in scripts:
        shutil.copyfile(s, prot / s.name)
    (prot / "run.sh").write_text("#!/bin/bash\necho my own script\n")
    xdg = tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield base, root, xdg, scripts
    proc.terminate()
    proc.wait(timeout=15)


def _session(base, path="structures/MgO.cif"):
    st, _, r = U.post(base, "api/edit/open", {"path": path})
    assert st == 200, r
    return r


# ------------------------------------------------------------------ C19-a
def test_every_protected_kind_is_refused(served):
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    base, root, xdg, scripts = served
    r = _session(base)
    names = [p.name for p in (root / "prot").iterdir() if p.name != "run.sh"]
    assert len(names) >= 20
    for name in names:
        target = root / "prot" / name
        before = sha(target)
        st, _, out = U.post(base, "api/structure/save", {"session": r["session"], "path": f"prot/{name}",
                                                         "format": "cif", "overwrite": True})
        assert st == 409 and out.get("protected"), (name, st, out)
        assert sha(target) == before, name
        st, _, c = U.req(base, "api/write/check?path=" + f"prot/{name}")
        assert st == 200 and c["protected"], name
    # MACE's own job scripts are recognized by content, a user's own script is an ordinary file
    from mace.gui import saving
    for s in scripts:
        assert saving.is_job_script(root / "prot" / s.name), s.name
    assert saving.protected_reason(root / "prot" / "run.sh") is None
    # the routes whose names cannot name a protected kind refuse by their own rules and write nothing
    for route, body in (("deck/save", {"name": "../prot/run.out", "mode": "cif2d12", "json": {}, "overwrite": True}),
                        ("setup/save", {"source": {"path": "prot/run.out"}, "json_text": "{}", "name": "run.out"})):
        st, _, out = U.post(base, f"api/{route}", body)
        assert st in (400, 403, 409), (route, st, out)
    assert sha(root / "prot" / "run.out") == sha(H.EXAMPLES / "outputs" / "Si_sp.out")


# ------------------------------------------------------------------ C19-b (tracked rows; SLURM via jobs.py lands with PLAN_submit)
def test_deck_of_a_queued_job_is_refused_until_the_job_ends(served):
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    base, root, _, _ = served
    d = root / "queued"
    d.mkdir(exist_ok=True)
    deck = d / "mat1_opt.d12"
    deck.write_text("mat1_opt\nCRYSTAL\nEND\n")
    db = d / "materials.db"
    c = sqlite3.connect(str(db))
    c.execute("CREATE TABLE calculations (calc_id TEXT, input_file TEXT, status TEXT, slurm_job_id TEXT)")
    c.execute("INSERT INTO calculations VALUES ('c1', ?, 'submitted', '900001')", (str(deck.resolve()),))
    c.commit()
    c.close()
    r = _session(base)
    st, _, out = U.post(base, "api/structure/save", {"session": r["session"], "path": "queued/mat1_opt.d12", "overwrite": True})
    assert st == 409 and "mat1_opt is queued; it reads this deck when it starts" in out["error"], out
    c = sqlite3.connect(str(db))
    c.execute("UPDATE calculations SET status = 'completed'")
    c.commit()
    c.close()
    st, _, out = U.post(base, "api/structure/save", {"session": r["session"], "path": "queued/mat1_opt.d12", "overwrite": True})
    assert st == 400 and "CIF" in out["error"]          # no longer protected: only the name rule answers now


# ------------------------------------------------------------------ C19-c (unit: the stamp is checked again before the rename)
def test_change_between_check_and_rename_is_a_conflict(tmp_path, monkeypatch):
    from mace.gui import saving, history
    history.configure(tmp_path)
    target = tmp_path / "a.cif"
    target.write_text("old\n")
    stamp = saving.file_stamp(target)
    real = saving._write_bytes

    def racing(fh, data):
        real(fh, data)
        time.sleep(0.01)
        target.write_text("written by someone else meanwhile\n")
    monkeypatch.setattr(saving, "_write_bytes", racing)
    with pytest.raises(saving.SaveConflict):
        saving.write_atomic(target, b"mine\n", expected=stamp)
    assert target.read_text() == "written by someone else meanwhile\n"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    history.configure(None)


# ------------------------------------------------------------------ C20-a, C20-b (the routes keep the old bytes first)
def _entries(base, rel):
    st, _, r = U.req(base, "api/history?path=" + rel)
    assert st == 200, r
    return r


def test_replacing_a_file_keeps_exactly_one_version(served):
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    base, root, xdg, _ = served
    # W1: structure/save over an existing CIF
    target = root / "structures" / "NaCl.cif"
    old = target.read_bytes()
    r = _session(base, "structures/NaCl.cif")
    U.post(base, "api/edit/op", {"session": r["session"], "op": "replace", "args": {"indices": [0], "element": "K"}})
    n0 = len(_entries(base, "structures/NaCl.cif")["entries"])
    st, _, out = U.post(base, "api/structure/save", {"session": r["session"], "path": "structures/NaCl.cif",
                                                     "overwrite": True, "expected": r["file"]})
    assert st == 200 and out["history"]["sha256"] == hashlib.sha256(old).hexdigest(), out
    h = _entries(base, "structures/NaCl.cif")
    assert len(h["entries"]) == n0 + 1
    st, _, data = U.req(base, "api/history/file?id=" + h["entries"][0]["id"])
    assert st == 200 and hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest() == hashlib.sha256(old).hexdigest()
    assert (root / "structures" / "NaCl.cif.bak").read_bytes() == old          # .bak: the oldest original, unchanged rule
    # W2: the classic panel's save_cif over an existing file
    (root / "w2.cif").write_bytes(old)
    st, _, out = U.post(base, "api/edit/save_cif", {"session": r["session"], "name": "w2.cif", "overwrite": True})
    assert st == 200 and out["history"]["sha256"] == hashlib.sha256(old).hexdigest(), out
    # W4: deck/save over an existing JSON
    j = root / "w4.json"
    j.write_text('{"calculation_type": "SP"}\n')
    oldj = j.read_bytes()
    st, _, out = U.post(base, "api/deck/save", {"name": "w4.json", "mode": "cif2d12", "json": {}, "overwrite": True})
    if st == 200:
        hh = _entries(base, "w4.json")["entries"]
        assert hh and hh[0]["sha256"] == hashlib.sha256(oldj).hexdigest()
    # a new file is listed as created by MACE (restore = delete it while it is unchanged)
    created = [e for e in _entries(base, "w2.cif")["entries"] if e.get("created")]
    assert not created          # w2.cif existed: a replace, not a creation
    # the index never records the folder's absolute path or file contents
    for idx in (xdg / "state").rglob("index.jsonl"):
        text = idx.read_text()
        assert str(root) not in text and "data_" not in text


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_history_store_not_writable_refuses_the_write(served):
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    base, root, xdg, _ = served
    store = xdg / "state" / "mace" / "gui" / "history"
    store.mkdir(parents=True, exist_ok=True)
    target = root / "structures" / "LiF.cif"
    before = sha(target)
    r = _session(base, "structures/LiF.cif")
    U.post(base, "api/edit/op", {"session": r["session"], "op": "replace", "args": {"indices": [0], "element": "Na"}})
    for d in [store, *[p for p in store.rglob("*") if p.is_dir()]]:
        d.chmod(0o500)
    try:
        st, _, out = U.post(base, "api/structure/save", {"session": r["session"], "path": "structures/LiF.cif",
                                                         "overwrite": True, "expected": r["file"]})
        assert st == 409 and "could not keep the previous version" in out["error"], out
        assert sha(target) == before
    finally:
        for d in [store, *[p for p in store.rglob("*") if p.is_dir()]]:
            d.chmod(0o700)


# ------------------------------------------------------------------ C20-c (retention, unit)
def test_retention_pools_and_the_newest_copy(tmp_path, monkeypatch):
    from mace.gui import history
    monkeypatch.setenv("MACE_GUI_HISTORY_DIR", str(tmp_path / "store"))
    root = tmp_path / "root"
    root.mkdir()
    history.configure(root)
    try:
        f = root / "a.d12"
        for k in range(25):
            f.write_text(f"version {k}\n" * 10)
            history.keep(f, f"save {k}")
        assert len(history.entries("a.d12")) == 20
        # entries older than historyDays go, except the newest
        s = {"files.historyKeep": 20, "files.historyDays": 1, "files.historyMaxMB": 500, "files.historyOutputsMaxMB": 1000}
        history.prune(s, now=time.time() + 3 * 86400)
        assert len(history.entries("a.d12")) == 1
        # the output pool never evicts input entries
        decks = []
        for k in range(5):
            dk = root / f"d{k}.d12"
            dk.write_text("x" * 2000)
            history.keep(dk, "input")
            history.keep(dk, "input again")
            decks.append(dk)
        out = root / "big.out"
        for k in range(6):
            out.write_bytes(os.urandom(300_000))
            history.keep(out, "Overwrite finished results", kind="output")
        s = {"files.historyKeep": 20, "files.historyDays": 30, "files.historyMaxMB": 500, "files.historyOutputsMaxMB": 1}
        s["files.historyOutputsMaxMB"] = 0.5
        history.prune(s)
        for dk in decks:
            assert len(history.entries(dk.name)) == 2
        outs = history.entries("big.out")
        assert 1 <= len(outs) < 6                       # pruned oldest first, the newest kept
        # a file over 50 MB is not copied
        monkeypatch.setattr(history, "MAX_FILE", 1000)
        e = history.keep(out, "too big", kind="output")
        assert e.skipped and "over 50 MB" in e.skipped
    finally:
        history.configure(None)


# ------------------------------------------------------------------ C20-h (route security)
def test_history_routes_refuse_escapes(served, tmp_path):
    base, root, xdg, _ = served
    for bad in ("../x", "%2e%2e/x", "/etc/passwd", "20260101T000000Z_1_a.cif%00", "20260101T000000Z_1_a.cif"):
        st, _, _ = U.req(base, "api/history/file?id=" + bad)
        assert st in (400, 404), bad
    # a symlink planted in the store is never followed
    store = xdg / "state" / "mace" / "gui" / "history"
    rh = next((p for p in store.iterdir() if p.is_dir()), None) if store.is_dir() else None
    if rh is not None:
        fd = next(p for p in rh.iterdir() if p.is_dir())
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        os.symlink(secret, fd / "20260101T000000Z_1_secret.txt")
        st, _, out = U.req(base, "api/history/file?id=20260101T000000Z_1_secret.txt")
        assert st == 404 and out != b"secret"
    for route, data in (("history", {"path": "../outside.cif"}),):
        st, _, _ = U.req(base, f"api/{route}?path=" + data["path"])
        assert st in (400, 403, 404)
    # restore onto a protected kind: refused; with a stale stamp: refused, current file unchanged
    st, _, out = U.post(base, "api/history/restore", {"id": "20260101T000000Z_1_x.cif", "path": "prot/run.out"})
    assert st in (404, 409)
    h = _entries(base, "structures/NaCl.cif")
    if h["entries"]:
        cur = sha(root / "structures" / "NaCl.cif")
        st, _, out = U.post(base, "api/history/restore", {"id": h["entries"][0]["id"], "path": "structures/NaCl.cif",
                                                          "expected": {"mtime_ns": "1", "size": 1}})
        assert st == 409 and sha(root / "structures" / "NaCl.cif") == cur
        # and with the current stamp it restores (the current version is kept first)
        n = len(h["entries"])
        st, _, out = U.post(base, "api/history/restore", {"id": h["entries"][0]["id"], "path": "structures/NaCl.cif",
                                                          "expected": h["file"]})
        assert st == 200, out
        assert len(_entries(base, "structures/NaCl.cif")["entries"]) == n + 1


def test_a_file_mace_created_can_be_deleted_from_file_history(served):
    """A save that creates a new file lists it as "created by MACE"; its
    restore deletes that file while its bytes are unchanged, and refuses
    once the file changed."""
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    base, root, xdg, _ = served
    r = _session(base, "structures/MgO.cif")
    for name in ("made_by_gui.cif", "made_then_edited.cif"):
        st, _, out = U.post(base, "api/structure/save", {"session": r["session"], "path": name, "expected": None})
        assert st == 200 and (root / name).is_file(), out
    h = _entries(base, "made_by_gui.cif")
    created = [e for e in h["entries"] if e.get("created")]
    assert len(created) == 1
    st, _, out = U.post(base, "api/history/restore", {"id": created[0]["id"], "path": "made_by_gui.cif",
                                                      "expected": h["file"]})
    assert st == 200 and out["deleted"] is True, out
    assert not (root / "made_by_gui.cif").exists()
    # changed after MACE created it: not deleted
    h = _entries(base, "made_then_edited.cif")
    created = [e for e in h["entries"] if e.get("created")]
    with open(root / "made_then_edited.cif", "a") as fh:
        fh.write("# my edit\n")
    h2 = _entries(base, "made_then_edited.cif")
    st, _, out = U.post(base, "api/history/restore", {"id": created[0]["id"], "path": "made_then_edited.cif",
                                                      "expected": h2["file"]})
    assert st == 409 and (root / "made_then_edited.cif").is_file(), out


def test_entries_of_another_folder_are_invisible(tmp_path):
    """An entry made while serving folder A is 404 when the server serves folder B."""
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    a, b = H.examples_copy(tmp_path / "A"), H.examples_copy(tmp_path / "B")
    xdg = tmp_path / "xdg"
    proc, base, _ = H.start_server(a, xdg, MACE_GUI_NO_WARMUP="1")
    try:
        st, _, r = U.post(base, "api/edit/open", {"path": "structures/MgO.cif"})
        st, _, out = U.post(base, "api/structure/save", {"session": r["session"], "path": "structures/MgO.cif",
                                                         "overwrite": True, "expected": r["file"]})
        eid = out["history"]["id"]
        assert U.req(base, "api/history/file?id=" + eid)[0] == 200
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    proc, base, _ = H.start_server(b, xdg, MACE_GUI_NO_WARMUP="1")
    try:
        assert U.req(base, "api/history/file?id=" + eid)[0] == 404
        assert _entries(base, "structures/MgO.cif")["entries"] == []
    finally:
        proc.terminate()
        proc.wait(timeout=15)


# ------------------------------------------------------------------ C20-i (the snapshot API, all or nothing)
@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_snapshot_all_or_nothing(tmp_path, monkeypatch):
    from mace.gui import history
    store = tmp_path / "store"
    monkeypatch.setenv("MACE_GUI_HISTORY_DIR", str(store))
    root = tmp_path / "root"
    root.mkdir()
    history.configure(root)
    try:
        a, p = root / "a.sh", root / "plan.json"
        a.write_text("#!/bin/bash\n")
        p.write_text("{}\n")
        got = history.snapshot([a, p], reason="Submit: test", route="submit")
        assert [e.rel_path for e in got] == ["a.sh", "plan.json"]
        o = root / "x.out"
        o.write_text("out\n")
        e = history.keep(o, "Overwrite finished results", kind="output")
        assert e.kind == "output"
        n_before = sum(1 for _ in store.rglob("*") if _.is_file())
        for d in [store, *[q for q in store.rglob("*") if q.is_dir()]]:
            d.chmod(0o500)
        try:
            with pytest.raises(history.HistoryUnavailable):
                history.snapshot([a, p], reason="Submit: again", route="submit")
        finally:
            for d in [store, *[q for q in store.rglob("*") if q.is_dir()]]:
                d.chmod(0o700)
        assert sum(1 for _ in store.rglob("*") if _.is_file()) == n_before
    finally:
        history.configure(None)
