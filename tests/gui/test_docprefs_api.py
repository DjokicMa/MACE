"""Per-document view options on the server (polish A3, K1), through the
real `mace gui`: $XDG_CONFIG_HOME/mace/gui-docprefs.json, 0600, inside the
served folder only, checked per key, 300 documents per kind, a corrupt file
set aside, and the token rule. Two servers writing at once keep both."""
import json
import os
import threading

import pytest

import guiharness as H
from test_shell_api import req

pytestmark = []


@pytest.fixture
def served(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    yield base, root, xdg
    proc.terminate()
    proc.wait(timeout=15)


def get(base, kind, path):
    from urllib.parse import urlencode
    return req(base, "api/docprefs?" + urlencode({"kind": kind, "path": path}))


def post(base, kind, path, values):
    return req(base, "api/docprefs", data={"kind": kind, "path": path, "values": values})


def test_round_trip_and_delete(served):
    base, root, xdg = served
    before = H.dir_md5(root)
    st, _, body = get(base, "display", "structures/MgO.cif")
    assert st == 200 and json.loads(body)["values"] == {}
    st, _, body = post(base, "display", "structures/MgO.cif", {"displayHi": [3, 3, 3], "showBoundary": False})
    assert st == 200, body
    assert json.loads(get(base, "display", "structures/MgO.cif")[2])["values"] == {"displayHi": [3, 3, 3], "showBoundary": False}
    f = xdg / "mace" / "gui-docprefs.json"
    assert f.is_file() and (f.stat().st_mode & 0o777) == 0o600
    assert H.dir_md5(root) == before                 # nothing in the served folder
    post(base, "display", "structures/MgO.cif", {})
    assert json.loads(get(base, "display", "structures/MgO.cif")[2])["values"] == {}


def test_refusals(served):
    base = served[0]
    assert post(base, "display", "../outside.cif", {"atomScale": 0.5})[0] == 400
    assert post(base, "display", "/etc/passwd", {"atomScale": 0.5})[0] == 400
    # a name that is no document makes no entry (the literal "%2e%2e" is a name, not "..")
    assert post(base, "display", "%2e%2e/outside/secret.cif", {"atomScale": 0.5})[0] == 400
    assert post(base, "display", "structures/none.cif", {"atomScale": 0.5})[0] == 400
    st, _, body = post(base, "display", "structures/MgO.cif", {"depthCue": 3})
    assert st == 400 and b"depthCue" in body
    assert post(base, "nope", "structures/MgO.cif", {})[0] == 400
    assert req(base, "api/docprefs?kind=display&path=structures/MgO.cif", token=False)[0] == 403


def test_lru_keeps_300_per_kind(served, tmp_path):
    from mace.gui import docprefs
    xdg = served[2]
    old = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(xdg)
    try:
        for k in range(301):
            docprefs.put("display", "h", f"d{k}.cif", {"atomScale": 0.5}, now=1000 + k)
        kinds = json.loads((xdg / "mace" / "gui-docprefs.json").read_text())["kinds"]
        docs = kinds["display"]["h"]
        assert len(docs) == 300 and "d0.cif" not in docs and "d300.cif" in docs
    finally:
        if old is None:
            os.environ.pop("XDG_CONFIG_HOME")
        else:
            os.environ["XDG_CONFIG_HOME"] = old


def test_corrupt_file_is_set_aside(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    (xdg / "mace").mkdir(parents=True)
    (xdg / "mace" / "gui-docprefs.json").write_text("{not json")
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    try:
        assert json.loads(get(base, "display", "structures/MgO.cif")[2])["values"] == {}
        assert (xdg / "mace" / "gui-docprefs.json.bad").read_text() == "{not json"
        assert post(base, "display", "structures/MgO.cif", {"atomScale": 0.6})[0] == 200
    finally:
        proc.terminate()
        proc.wait(timeout=15)


@pytest.mark.parametrize("stored", [{"kinds": {"display": []}}, {"kinds": [1]},
                                    {"kinds": {"display": {"h": {"a.cif": "x"}}}},
                                    {"kinds": {"display": {"h": []}}}])
def test_wrong_shaped_file_is_set_aside(tmp_path, stored):
    """JSON of the wrong shape (a hand edit) is set aside like unreadable JSON,
    instead of failing every read and write of per-document options."""
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    (xdg / "mace").mkdir(parents=True)
    (xdg / "mace" / "gui-docprefs.json").write_text(json.dumps(stored))
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    try:
        assert json.loads(get(base, "display", "structures/MgO.cif")[2])["values"] == {}
        assert json.loads((xdg / "mace" / "gui-docprefs.json.bad").read_text()) == stored
        assert post(base, "display", "structures/MgO.cif", {"atomScale": 0.6})[0] == 200
        assert json.loads(get(base, "display", "structures/MgO.cif")[2])["values"] == {"atomScale": 0.6}
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_two_servers_patch_at_once(tmp_path):
    """Two servers on one XDG dir, 50 writes each from two threads: every
    document (and both keys of a shared one) is kept."""
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    mgo = (root / "structures" / "MgO.cif").read_bytes()
    for tag in "ab":
        for k in range(50):
            (root / f"doc_{tag}_{k}.cif").write_bytes(mgo)
    a, base_a, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    b, base_b, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    errors = []

    def run(base, tag):
        for k in range(50):
            st, _, body = post(base, "display", f"doc_{tag}_{k}.cif", {"atomScale": 0.5})
            if st != 200:
                errors.append((st, body))

    try:
        ts = [threading.Thread(target=run, args=(base_a, "a")), threading.Thread(target=run, args=(base_b, "b"))]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert not errors, errors[:3]
        data = json.loads((xdg / "mace" / "gui-docprefs.json").read_text())
        docs = next(iter(data["kinds"]["display"].values()))
        assert len(docs) == 100 and data["rev"] == 100
    finally:
        for p in (a, b):
            p.terminate()
            p.wait(timeout=15)
