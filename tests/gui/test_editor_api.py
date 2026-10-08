"""The editor's server side through the real `mace gui` (spec §1.8, Phase 3):
/api/symmetry, the first edit opening its session, revisions and jumps,
sessions from a held structure, and the one save path with its rules
(overwrite, changed on disk, MgO.cif.bak, atomic writes, inside --dir)."""
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import guiharness as H

pytest.importorskip("ase")
pytest.importorskip("spglib")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("edapi")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1",
                                   MACE_GUI_MAX_WORKERS="2")
    yield base, root, xdg
    proc.terminate()
    proc.wait(timeout=15)


def req(base, path, data=None, headers=None, token=True, method=None):
    h = {"X-Requested-With": "mace-gui"}
    if token:
        h["X-Mace-Token"] = H.TOKEN
    h.update(headers or {})
    body = json.dumps(data).encode() if data is not None else None
    if body is not None:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(base + path, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, dict(e.headers), json.loads(raw)
        except ValueError:
            return e.code, dict(e.headers), raw


def post(base, path, data, **kw):
    return req(base, path, data=data, **kw)


def tree_md5(root: Path):
    h = hashlib.md5()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def structure(base, path):
    st, _, s = req(base, "api/structure?path=" + path)
    assert st == 200, s
    return s


def mgo_body(base, **extra):
    s = structure(base, "structures/MgO.cif")
    return {"cell": s["cell"], "numbers": s["numbers"], "frac": s["frac"], **extra}


# --- /api/symmetry -------------------------------------------------------------------

def test_symmetry_mgo_225_equivalents(served):
    base, root, _ = served
    before = tree_md5(root)
    st, headers, r = post(base, "api/symmetry", mgo_body(base, symprec=1e-3, ops=True, key="doc1"))
    assert st == 200, r
    assert r["number"] == 225 and r["equivalent_atoms"] == [0, 0, 0, 0, 4, 4, 4, 4]
    assert r["wyckoffs"] == ["a"] * 4 + ["b"] * 4 and len(r["site_symmetry"]) == 8
    assert r["multiplicity"] == [4] * 8 and r["n_operations_conventional"] == 192
    assert len(r["rotations"]) == 192 and len(r["translations"]) == 192
    assert "symm;dur=" in headers["Server-Timing"] and "wait;dur=" in headers["Server-Timing"]
    assert tree_md5(root) == before                       # a read never writes


def test_symmetry_route_rejects_bad_input(served):
    base = served[0]
    good = mgo_body(base)
    big = {"cell": [[100, 0, 0], [0, 100, 0], [0, 0, 100]], "numbers": [1] * 5001,
           "frac": [[i / 5001, 0, 0] for i in range(5001)]}
    assert post(base, "api/symmetry", big)[0] == 400
    assert post(base, "api/symmetry", {**good, "symprec": float("nan")})[0] == 400
    assert post(base, "api/symmetry", {**good, "symprec": -1})[0] == 400
    st, _, r = post(base, "api/symmetry", {"cell": good["cell"], "frac": good["frac"]})
    assert st == 400 and "numbers" in r["error"]
    assert post(base, "api/symmetry", {**good, "frac": good["frac"][:3]})[0] == 400
    assert post(base, "api/symmetry", {**good, "numbers": [0] * 8})[0] == 400
    # an atomic number past the table is refused, never folded onto another element
    assert post(base, "api/symmetry", {**good, "numbers": [150] * 4 + [8] * 4})[0] == 400


def test_symmetry_coalesces_per_key_and_respects_cap(served):
    """With spglib held 200 ms per call: ten quick calls for one document run
    spglib at most twice, the rest are superseded; calls for cap+2 documents
    never run more than the cap (2 here) at once."""
    base = served[0]
    body = mgo_body(base)
    post(base, "api/test/gate/reset", {})
    out = []

    def one(key, hold):
        out.append(post(base, "api/symmetry", {**body, "key": key}, headers={"X-Mace-Test-Hold-Ms": str(hold)})[2])

    threads = []
    for k in range(10):
        t = threading.Thread(target=one, args=("same-doc", 200))
        t.start()
        threads.append(t)
        time.sleep(0.01)
    for t in threads:
        t.join()
    ran = [r for r in out if not r.get("superseded")]
    assert len(out) == 10 and 1 <= len(ran) <= 2, out
    assert all(r["number"] == 225 for r in ran)
    gate = req(base, "api/test/gate")[2]
    assert gate["superseded"] >= 8 and gate["cap"] == 2
    # cap + 2 different documents: never more than the cap at once
    post(base, "api/test/gate/reset", {})
    out.clear()
    threads = [threading.Thread(target=one, args=(f"doc{k}", 200)) for k in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    gate = req(base, "api/test/gate")[2]
    assert gate["max_running"] <= 2 and gate["ran"] == 4 and not any(r.get("superseded") for r in out)


# --- sessions -------------------------------------------------------------------------

def test_first_op_opens_session_in_one_request(served):
    base, root, _ = served
    s = structure(base, "structures/MgO.cif")
    st, _, r = post(base, "api/edit/op", {"open": {"path": "structures/MgO.cif", "locked": True,
                                                   "symprec": 1e-3, "expected": s["file"]},
                                          "op": "replace", "args": {"indices": [0], "element": "Ca"}})
    assert st == 200, r
    assert r["session"] and r["symbols"].count("Ca") == 4 and r["locked"]
    assert [h["label"] for h in r["history"]] == ["Opened MgO.cif", "Replace Mg→Ca (4 atoms, orbit 4a)"]
    assert r["opened_rev"] == r["history"][0]["rev"] and r["file"] == s["file"]
    # a stale stamp (the file changed on disk since it was opened) opens nothing
    st, _, r = post(base, "api/edit/op", {"open": {"path": "structures/MgO.cif",
                                                   "expected": {"mtime_ns": "1", "size": 1}},
                                          "op": "delete", "args": {"indices": [0]}})
    assert st == 409 and r["changed_on_disk"]


def test_open_keeps_atom_order(served):
    base = served[0]
    for path in ["structures/MgO.cif", "structures/NaCl.cif", "structures/Si.cif", "structures/LiF.cif",
                 "structures/diamond.cif", "outputs/Si_sp.out", "outputs/Ag2Br3_opt_unfinished.out"]:
        s = structure(base, path)
        st, _, e = post(base, "api/edit/open", {"path": path})
        assert st == 200, (path, e)
        assert e["numbers"] == [n % 100 for n in s["numbers"]], path
        assert all(abs(a - b) < 1e-9 for fa, fb in zip(e["frac"], s["frac"]) for a, b in zip(fa, fb)), path
        assert e["file"] == s["file"]


def test_edit_jump_and_rev_round_trip(served):
    base = served[0]
    st, _, r = post(base, "api/edit/open", {"path": "structures/NaCl.cif"})
    sid, r0 = r["session"], r["rev"]
    _, _, r1 = post(base, "api/edit/op", {"session": sid, "op": "delete", "args": {"indices": [0]}, "base_rev": r0})
    _, _, r2 = post(base, "api/edit/op", {"session": sid, "op": "delete", "args": {"indices": [0]},
                                          "base_rev": r1["rev"]})
    st, _, j = post(base, "api/edit/jump", {"session": sid, "index": 0, "base_rev": r2["rev"]})
    assert st == 200 and j["rev"] == r0 and j["position"] == 0 and len(j["numbers"]) == 8
    assert [h["rev"] for h in j["history"]] == [r0, r1["rev"], r2["rev"]]
    st, _, j = post(base, "api/edit/jump", {"session": sid, "index": 2, "base_rev": r0})
    assert st == 200 and j["rev"] == r2["rev"] and len(j["numbers"]) == 6


def test_stale_base_rev_is_409_and_applies_nothing(served):
    base = served[0]
    _, _, r = post(base, "api/edit/open", {"path": "structures/MgO.cif"})
    sid, r0 = r["session"], r["rev"]
    _, _, r1 = post(base, "api/edit/op", {"session": sid, "op": "replace",
                                          "args": {"indices": [0], "element": "Ca"}, "base_rev": r0})
    st, _, c = post(base, "api/edit/op", {"session": sid, "op": "delete", "args": {"indices": [1]}, "base_rev": r0})
    assert st == 409 and c["conflict"] == "rev" and c["rev"] == r1["rev"]
    assert len(c["payload"]["numbers"]) == 8 and c["payload"]["rev"] == r1["rev"]
    st, _, s = req(base, f"api/edit/state?session={sid}")
    assert len(s["numbers"]) == 8 and s["rev"] == r1["rev"]


def test_reversed_arrival_is_refused_not_reordered(served):
    """Two edits sent at once, the first held back on the server: the second
    carries the first's revision as its base, so it can only be refused (409),
    never applied before the first. The browser's queue never sends it early."""
    base = served[0]
    _, _, r = post(base, "api/edit/open", {"path": "structures/MgO.cif"})
    sid, r0 = r["session"], r["rev"]
    res = {}

    def first():
        res["a"] = post(base, "api/edit/op", {"session": sid, "op": "replace", "base_rev": r0,
                                              "args": {"indices": [0], "element": "Ca"}},
                        headers={"X-Mace-Test-Delay-Ms": "300", "X-Mace-Test-Tag": "a"})

    t = threading.Thread(target=first)
    t.start()
    time.sleep(0.05)
    res["b"] = post(base, "api/edit/op", {"session": sid, "op": "replace", "base_rev": r0 + 999,
                                          "args": {"indices": [1], "element": "Sr"}},
                    headers={"X-Mace-Test-Tag": "b"})
    t.join()
    assert res["a"][0] == 200 and res["b"][0] == 409
    st, _, s = req(base, f"api/edit/state?session={sid}")
    assert s["symbols"].count("Sr") == 0 and s["symbols"][0] == "Ca"


def test_from_structure_session_matches_payload(served):
    base = served[0]
    s = structure(base, "structures/LiF.cif")
    st, _, r = post(base, "api/edit/from_structure", {"name": "LiF.cif", "cell": s["cell"], "numbers": s["numbers"],
                                                       "frac": s["frac"], "locked": True})
    assert st == 200, r
    assert r["numbers"] == s["numbers"] and r["locked"] and r["history"][0]["label"] == "Restored LiF.cif"
    assert r["symmetry"]["number"] == s["symmetry"]["number"]
    st, _, r = post(base, "api/edit/from_structure", {"name": "x", "cell": s["cell"], "numbers": [999],
                                                       "frac": [[0, 0, 0]]})
    assert st == 400
    st, _, r = post(base, "api/edit/from_structure", {"name": "../x", "cell": s["cell"],
                                                       "numbers": s["numbers"], "frac": s["frac"]})
    assert st == 400


def test_evicted_session_is_404(served):
    base = served[0]
    _, _, r = post(base, "api/edit/open", {"path": "structures/MgO.cif"})
    assert post(base, "api/test/evict", {"session": r["session"]})[2]["evicted"]
    st, _, e = post(base, "api/edit/op", {"session": r["session"], "op": "delete", "args": {"indices": [0]}})
    assert st == 404


# --- saving --------------------------------------------------------------------------------

def _session(base, path="structures/MgO.cif", **extra):
    _, _, r = post(base, "api/edit/open", {"path": path, **extra})
    return r


def test_save_refuses_overwrite_without_flag(served):
    base, root, _ = served
    r = _session(base)
    st, _, out = post(base, "api/structure/save", {"session": r["session"], "path": "structures/MgO.cif"})
    assert st == 409 and out["needs_overwrite"]
    st, _, out = post(base, "api/structure/save", {"session": r["session"], "path": "MgO_new.cif"})
    assert st == 200 and out["saved"] == "MgO_new.cif" and out["bak"] is None
    assert (root / "MgO_new.cif").read_text().startswith("data_")
    (root / "MgO_new.cif").unlink()


def test_save_makes_bak_once_and_keeps_original(served):
    base, root, _ = served
    target = root / "structures" / "NaCl.cif"
    original = target.read_bytes()
    r = _session(base, "structures/NaCl.cif")
    post(base, "api/edit/op", {"session": r["session"], "op": "replace", "args": {"indices": [0], "element": "K"}})
    st, _, out = post(base, "api/structure/save", {"session": r["session"], "path": "structures/NaCl.cif",
                                                   "overwrite": True, "expected": r["file"], "backup": True})
    assert st == 200 and out["bak"] == "structures/NaCl.cif.bak" and out["bak_created"]
    assert out["file"]["size"] == target.stat().st_size and b"K" in target.read_bytes()
    post(base, "api/edit/op", {"session": r["session"], "op": "replace", "args": {"indices": [1], "element": "Rb"}})
    st, _, out2 = post(base, "api/structure/save", {"session": r["session"], "path": "structures/NaCl.cif",
                                                    "overwrite": True, "expected": out["file"]})
    assert st == 200 and out2["bak_created"] is False
    bak = root / "structures" / "NaCl.cif.bak"
    assert hashlib.md5(bak.read_bytes()).hexdigest() == hashlib.md5(original).hexdigest()
    bak.unlink()
    target.write_bytes(original)


def test_save_never_overwrites_existing_bak(served):
    base, root, _ = served
    target = root / "structures" / "LiF.cif"
    original = target.read_bytes()
    bak = root / "structures" / "LiF.cif.bak"
    bak.write_text("an older backup\n")
    r = _session(base, "structures/LiF.cif")
    st, _, out = post(base, "api/structure/save", {"session": r["session"], "path": "structures/LiF.cif",
                                                   "overwrite": True, "expected": r["file"]})
    assert st == 200 and out["bak"] == "structures/LiF.cif.bak" and out["bak_created"] is False
    assert bak.read_text() == "an older backup\n" and out["bak_t"]
    bak.unlink()
    target.write_bytes(original)


def test_save_changed_on_disk_is_409(served):
    base, root, _ = served
    target = root / "structures" / "Si.cif"
    original = target.read_bytes()
    r = _session(base, "structures/Si.cif")
    time.sleep(0.02)
    target.write_bytes(original + b"\n# touched by another program\n")
    st, _, out = post(base, "api/structure/save", {"session": r["session"], "path": "structures/Si.cif",
                                                   "overwrite": True, "expected": r["file"]})
    assert st == 409 and out["changed_on_disk"] and out["size"] == target.stat().st_size
    assert target.read_bytes().endswith(b"another program\n")             # nothing written
    st, _, out = post(base, "api/structure/save", {"session": r["session"], "path": "structures/Si.cif",
                                                   "overwrite": True, "backup": False,
                                                   "expected": {"mtime_ns": out["mtime_ns"], "size": out["size"]}})
    assert st == 200 and target.read_text().startswith("data_")
    assert not (root / "structures" / "Si.cif.bak").exists()
    target.write_bytes(original)


def test_save_refuses_outside_dir(served, tmp_path):
    base, root, _ = served
    r = _session(base)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    link = root / "out_link"
    link.symlink_to(outside, target_is_directory=True)
    try:
        for path in ["../x.cif", str(tmp_path / "abs.cif"), "out_link/x.cif"]:
            st, _, out = post(base, "api/structure/save", {"session": r["session"], "path": path})
            assert st == 403, (path, out)
        assert not list(outside.iterdir()) and not (tmp_path / "abs.cif").exists()
        st, _, out = post(base, "api/structure/save", {"session": r["session"], "path": "nofolder/x.cif"})
        assert st == 400 and "does not exist" in out["error"]
        st, _, out = post(base, "api/structure/save", {"session": r["session"], "path": "x.txt"})
        assert st == 400 and ".cif" in out["error"]
    finally:
        link.unlink()


def test_save_poscar_and_xyz_round_trip(served):
    import ase.io
    import numpy as np
    base, root, _ = served
    r = _session(base)
    for fmt, name in [("poscar", "MgO.vasp"), ("xyz", "MgO.xyz"), ("poscar", "POSCAR")]:
        st, _, out = post(base, "api/structure/save", {"session": r["session"], "path": name, "format": fmt})
        assert st == 200, out
        back = ase.io.read(root / name, format="vasp" if fmt == "poscar" else "extxyz")
        assert back.get_chemical_symbols() == r["symbols"]
        assert np.allclose(back.cell.array, r["cell"], atol=1e-6)
        assert np.allclose(back.positions, r["cart"], atol=1e-5)
        (root / name).unlink()
    st, _, out = post(base, "api/structure/save", {"session": r["session"], "path": "MgO.cif", "format": "xyz"})
    assert st == 400


def test_save_atomic_no_partial_file(tmp_path, monkeypatch):
    """The writer fails half-way: the old file is untouched and no temporary file is left."""
    from ase.build import bulk
    from mace.gui import editor, saving, server
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    root = tmp_path / "root"
    root.mkdir()
    (root / "MgO.cif").write_text("the original\n")
    st = server.GuiState(root, "t")
    try:
        sess = editor.EditSession(bulk("MgO", "rocksalt", a=4.212, cubic=True), "MgO")

        def broken(fh, data):
            fh.write(data[: len(data) // 2])
            raise OSError("disk full")
        monkeypatch.setattr(saving, "_write_bytes", broken)
        with pytest.raises(OSError):
            saving.save(st, sess, "MgO.cif", overwrite=True, backup=False)
        assert (root / "MgO.cif").read_text() == "the original\n"
        assert sorted(p.name for p in root.iterdir()) == ["MgO.cif"]
    finally:
        st.close()


@pytest.mark.parametrize("path, data", [
    ("api/symmetry", {"cell": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "numbers": [1], "frac": [[0, 0, 0]]}),
    ("api/structure/save", {"session": "x", "path": "x.cif"}),
    ("api/edit/jump", {"session": "x", "index": 0}),
    ("api/edit/from_structure", {"name": "x"}),
    ("api/edit/op", {"open": {"path": "structures/MgO.cif"}, "op": "delete", "args": {"indices": [0]}}),
    ("api/structure/export", {"source": {"path": "structures/MgO.cif"}, "format": "cif"}),
])
def test_new_routes_require_token_host_and_xrw(served, path, data):
    base = served[0]
    assert post(base, path, data, token=False)[0] == 403
    port = base.split("/")[2].rsplit(":", 1)[1]
    assert post(base, path, data, headers={"Host": f"evil.example:{port}"})[0] == 421
    assert post(base, path, data, headers={"X-Requested-With": "x"})[0] == 403


def test_test_hooks_are_off_without_the_env(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    try:
        assert req(base, "api/test/gate")[0] == 404
        assert post(base, "api/test/evict", {"session": "x"})[0] == 404
        info = req(base, "api/info")[2]
        assert len(info["dir_hash"]) == 16
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_edit_close_lets_go_of_the_session(served):
    base, root, _ = served
    s = structure(base, "structures/MgO.cif")
    st, _, r = post(base, "api/edit/open", {"path": "structures/MgO.cif", "expected": s["file"]})
    assert st == 200
    sid = r["session"]
    assert post(base, "api/edit/close", {"session": sid})[2] == {"closed": True}
    assert req(base, f"api/edit/state?session={sid}")[0] == 404
    assert post(base, "api/edit/close", {"session": sid})[2] == {"closed": False}       # twice is harmless


def test_new_file_gets_the_usual_mode(served):
    """Found on HPCC: a saved new file was 0600 (mkstemp's mode), its neighbors 0640."""
    import stat
    base, root, _ = served
    s = structure(base, "structures/MgO.cif")
    st, _, r = post(base, "api/edit/open", {"path": "structures/MgO.cif", "expected": s["file"]})
    st, _, out = post(base, "api/structure/save", {"session": r["session"], "base_rev": r["rev"],
                                                    "path": "structures/MgO_mode.cif", "format": "cif"})
    assert st == 200, out
    mode = stat.S_IMODE((root / "structures" / "MgO_mode.cif").stat().st_mode)
    umask = os.umask(0)
    os.umask(umask)
    assert mode == 0o666 & ~umask, oct(mode)
    (root / "structures" / "MgO_mode.cif").unlink()
