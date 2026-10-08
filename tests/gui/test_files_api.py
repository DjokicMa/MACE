"""Files through the real `mace gui` (plan T3, B5, B7; spec sections 8.4.2,
8.4.5, 9.2): the parse cache (hit under 5 ms, keyed by mtime and size, an
ETag that answers 304), packed payloads, Export Structure as bytes that are
never written, the held store's pins and leases and its budget, and the
provenance sidecar that keeps "built from" across a save and a reopen,
outside the served folder."""
import base64
import hashlib
import io
import json
import os
import time
import warnings
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

ase_io = pytest.importorskip("ase.io")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("files")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2")
    yield base, root, xdg, proc
    proc.terminate()
    proc.wait(timeout=15)


def timing(headers):
    out = {}
    for part in (headers.get("Server-Timing") or "").split(","):
        name, _, rest = part.strip().partition(";")
        for kv in rest.split(";"):
            k, _, v = kv.partition("=")
            out[f"{name}.{k.strip()}"] = v.strip()
    return out


def test_parse_cache_hit_under_5ms(served):
    base = served[0]
    U.req(base, "api/structure?path=structures/Si.cif")
    for _ in range(3):
        st, h, r = U.req(base, "api/structure?path=structures/Si.cif")
        t = timing(h)
        assert st == 200 and t["cache.desc"] == "hit", h.get("Server-Timing")
        assert float(t["parse.dur"]) < 5, t
    assert r["symmetry"]["number"] == 227 and len(r["numbers"]) == 8


def test_parse_cache_invalidates_on_mtime(served):
    base, root = served[0], served[1]
    p = root / "structures" / "LiF.cif"
    st, h1, r1 = U.req(base, "api/structure?path=structures/LiF.cif")
    assert U.req(base, "api/structure?path=structures/LiF.cif")[1]["Server-Timing"].startswith("cache;desc=hit")
    now = time.time() + 5
    os.utime(p, (now, now))
    st, h2, r2 = U.req(base, "api/structure?path=structures/LiF.cif")
    assert timing(h2)["cache.desc"] == "miss" and h2["ETag"] != h1["ETag"]
    assert r2["file"]["mtime_ns"] != r1["file"]["mtime_ns"]


def test_structure_etag_304(served):
    base = served[0]
    st, h, _ = U.req(base, "api/structure?path=structures/MgO.cif&packed=1")
    st2, h2, body = U.req(base, "api/structure?path=structures/MgO.cif&packed=1", headers={"If-None-Match": h["ETag"]})
    assert st2 == 304 and not body and h2["ETag"] == h["ETag"]
    assert timing(h2)["cache.desc"] == "hit"


def test_packed_payload_exact_and_small(served):
    """The packed frac is the parsed one to the last bit; z, cell and the rest kept."""
    base = served[0]
    full = U.req(base, "api/structure?path=structures/NaCl.cif")[2]
    st, h, p = U.req(base, "api/structure?path=structures/NaCl.cif&packed=1", headers={"Accept-Encoding": "identity"})
    assert p["packed"] == 1 and p["n"] == len(full["numbers"]) and "symbols" not in p and "cart" not in p
    frac = np.frombuffer(base64.b64decode(p["frac"]), "<f8").reshape(-1, 3)
    assert frac.tobytes() == np.asarray(full["frac"], "<f8").tobytes()
    assert list(np.frombuffer(base64.b64decode(p["zn"]), np.uint8)) == full["numbers"]
    assert p["symmetry"] == full["symmetry"] and p["lattice"] == full["lattice"]


def test_files_list_has_stamps(served):
    base = served[0]
    files = U.req(base, "api/files")[2]["files"]
    f = [x for x in files if x["path"] == "structures/MgO.cif"][0]
    st = os.stat(served[1] / "structures" / "MgO.cif")
    assert f["mtime_ns"] == str(st.st_mtime_ns) and f["size"] == st.st_size


def export(base, source, fmt, mode="p1"):
    return U.post(base, "api/structure/export", {"source": source, "format": fmt, "cif_mode": mode})


def test_export_formats_and_no_write(served):
    base, root = served[0], served[1]
    before = U.tree_md5(root)
    ref = ase_io.read(str(H.EXAMPLES / "structures" / "MgO.cif"))
    for fmt, reader in (("cif", "cif"), ("poscar", "vasp"), ("xyz", "xyz"), ("extxyz", "extxyz")):
        st, h, data = export(base, {"path": "structures/MgO.cif"}, fmt)
        assert st == 200 and h["Content-Disposition"] == "attachment", (fmt, st, data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = ase_io.read(io.StringIO(data.decode()), format=reader)
        assert len(a) == len(ref) and sorted(a.numbers) == sorted(ref.numbers), fmt
        if fmt != "xyz":
            assert np.allclose(a.cell.array, ref.cell.array, atol=1e-6), fmt
        else:
            assert "Lattice" not in data.decode()                 # plain XYZ: no cell
    st, _, sym = export(base, {"path": "structures/MgO.cif"}, "cif", "symmetrized")
    assert b"Fm-3m" in sym.replace(b" ", b"") or b"225" in sym
    assert U.tree_md5(root) == before


def test_export_p1_equals_save_as_bytes(served):
    base, root = served[0], served[1]
    s = U.req(base, "api/structure?path=structures/MgO.cif")[2]
    st, _, r = U.post(base, "api/edit/open", {"path": "structures/MgO.cif", "expected": s["file"]})
    sid, rev = r["session"], r["rev"]
    st, _, ex = export(base, {"session": sid, "rev": rev}, "cif")
    st, _, out = U.post(base, "api/structure/save", {"session": sid, "base_rev": rev, "path": "structures/MgO_exportcheck.cif", "format": "cif"})
    assert st == 200, out
    saved = (root / "structures" / "MgO_exportcheck.cif").read_bytes()
    assert ex == saved
    (root / "structures" / "MgO_exportcheck.cif").unlink()
    # a stale revision is refused, as for every edit
    assert export(base, {"session": sid, "rev": rev - 1}, "cif")[0] == 409


def test_export_rejects_bad_requests(served):
    base = served[0]
    assert export(base, {"path": "structures/MgO.cif"}, "pdb")[0] == 400
    assert export(base, {"path": "structures/MgO.cif"}, "cif", "fancy")[0] == 400
    assert export(base, {"path": "../etc/passwd"}, "cif")[0] == 403
    assert export(base, {"held": "nope"}, "cif")[0] == 404
    assert export(base, "structures/MgO.cif", "cif")[0] == 400


def test_held_store_pinned_and_released(served):
    base = served[0]
    st, _, r = U.post(base, "api/test/held/put", {"atoms": 6000})
    hid = r["held"]
    rep = U.req(base, "api/test/held")[2]
    assert rep["pinned"] >= 1 and rep["pinned_bytes"] > 0
    st, _, p = U.req(base, f"api/held?id={hid}")
    assert st == 200 and p["packed"] == 1 and p["n"] == r["n"]
    st, _, d = export(base, {"held": hid}, "extxyz")
    assert st == 200 and d.decode().splitlines()[0].strip() == str(r["n"])
    assert U.post(base, "api/held/release", {"ids": [hid]})[2] == {"released": 1}
    assert U.req(base, f"api/held?id={hid}")[0] == 404


def test_held_lease_freed_after_reload_and_disconnect(served):
    """Two 6000-atom held structures leased to an event stream: when it closes
    (a reload, a crash) and nobody claims them, they unpin after the grace."""
    base = served[0]
    U.post(base, "api/test/held/config", {"grace_s": 1.0})
    sse = H.sse_events(base)
    try:
        hello = sse.wait(lambda e: e[0] == "hello", 10)[1]
        client = hello["client"]
        ids = [U.post(base, "api/test/held/put", {"atoms": 6000}, headers={"X-Mace-Client": client})[2]["held"] for _ in range(2)]
        assert U.req(base, "api/test/held")[2]["pinned"] >= 2
    finally:
        sse.close()
    end = time.time() + 10
    while time.time() < end and U.req(base, "api/resources")[2]["stores"]["held_pinned_mb"] > 0:
        time.sleep(0.2)
    assert U.req(base, "api/resources")[2]["stores"]["held_pinned_mb"] == 0
    # unpinned: evicted first when room is needed; a claim pins it again while it is still there
    sse2 = H.sse_events(base)
    try:
        c2 = sse2.wait(lambda e: e[0] == "hello", 10)[1]["client"]
        got = U.post(base, "api/held/claim", {"ids": ids}, headers={"X-Mace-Client": c2})[2]["claimed"]
        assert set(got) <= set(ids)
    finally:
        sse2.close()
    U.post(base, "api/held/release", {"ids": ids})
    U.post(base, "api/test/held/config", {"grace_s": 30.0})


def test_held_claim_within_grace_keeps_pin(served):
    base = served[0]
    U.post(base, "api/test/held/config", {"grace_s": 2.0})
    sse = H.sse_events(base)
    c1 = sse.wait(lambda e: e[0] == "hello", 10)[1]["client"]
    hid = U.post(base, "api/test/held/put", {"atoms": 800}, headers={"X-Mace-Client": c1})[2]["held"]
    sse.close()                                         # the reload
    sse2 = H.sse_events(base)
    try:
        c2 = sse2.wait(lambda e: e[0] == "hello", 10)[1]["client"]
        assert U.post(base, "api/held/claim", {"ids": [hid]}, headers={"X-Mace-Client": c2})[2]["claimed"] == [hid]
        time.sleep(3)                                   # past the old stream's grace
        assert U.req(base, f"api/held?id={hid}")[0] == 200
        assert U.req(base, "api/test/held")[2]["pinned"] >= 1
    finally:
        sse2.close()
        U.post(base, "api/held/release", {"ids": [hid]})
        U.post(base, "api/test/held/config", {"grace_s": 30.0})


def test_held_store_refuses_over_quarter_budget(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", MACE_GUI_CACHE_MB="64")
    try:
        got, refused = [], None
        for _ in range(12):
            st, _, r = U.post(base, "api/test/held/put", {"atoms": 60000})
            if st != 200:
                refused = r
                break
            got.append(r["held"])
        assert refused is not None and got, refused
        assert refused["error"].startswith("Close some of the big built structures first"), refused
        assert f"({len(got)} open, holding" in refused["error"]
        stores = U.req(base, "api/resources")[2]["stores"]
        assert stores["held_pinned_mb"] <= 16.0 + 1e-6 and stores["max_held_pinned_mb"] <= 16.0 + 1e-6, stores
        assert stores["total_mb"] <= stores["limit_mb"]
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_ledger_counts_parse_and_held(served):
    base = served[0]
    U.req(base, "api/structure?path=structures/hBN.cif")
    hid = U.post(base, "api/test/held/put", {"atoms": 2000})[2]["held"]
    stores = U.req(base, "api/resources")[2]["stores"]
    assert stores["parse_mb"] > 0 and stores["held_mb"] > 0 and stores["held_pinned_mb"] > 0, stores
    assert stores["held_pinned_cap_mb"] == stores["limit_mb"] / 4
    U.post(base, "api/held/release", {"ids": [hid]})


def sidecars(xdg):
    return sorted((Path(xdg) / "cache" / "mace" / "gui" / "provenance").glob("*.json"))


def test_provenance_survives_save_and_reopen(served):
    """A built document saved as a CIF: reopening the file brings back where it
    came from and its charge hint; editing the file outside drops them, quietly."""
    base, root, xdg = served[0], served[1], served[2]
    s = U.req(base, "api/structure?path=structures/MgO.cif")[2]
    meta = {"provenance": {"parent": "structures/MgO.cif", "op": "supercell", "params": {"n": [1, 1, 1]},
                           "label": "Supercell 1×1×1 of MgO.cif"}, "hints": {"charge": 2}}
    st, _, r = U.post(base, "api/edit/from_structure", {"name": "MgO_built", "cell": s["cell"], "frac": s["frac"],
                                                       "numbers": s["numbers"], "label": "Supercell 1×1×1 of MgO.cif", "meta": meta})
    assert st == 200 and r["doc_meta"]["hints"]["charge"] == 2
    before = U.tree_md5(root)
    st, _, out = U.post(base, "api/structure/save", {"session": r["session"], "base_rev": r["rev"],
                                                     "path": "structures/MgO_built.cif", "format": "cif"})
    assert st == 200, out
    saved = root / "structures" / "MgO_built.cif"
    # nothing but the CIF itself appeared in the folder; the sidecar is in the user's cache
    assert U.tree_md5(root, skip=("MgO_built.cif",)) == before
    assert any(p.stem == hashlib.sha256(saved.read_bytes()).hexdigest() for p in sidecars(xdg))
    again = U.req(base, "api/structure?path=structures/MgO_built.cif")[2]
    assert again["doc_meta"]["provenance"]["label"] == "Supercell 1×1×1 of MgO.cif" and again["doc_meta"]["hints"]["charge"] == 2
    st, _, sess = U.post(base, "api/edit/open", {"path": "structures/MgO_built.cif", "expected": again["file"]})
    assert sess["doc_meta"]["hints"]["charge"] == 2
    # edited outside the GUI: the hash no longer matches, the provenance is gone, no error
    saved.write_text(saved.read_text() + "\n# edited by hand\n")
    st, _, edited = U.req(base, "api/structure?path=structures/MgO_built.cif")
    assert st == 200 and "doc_meta" not in edited
    saved.unlink()


@pytest.mark.parametrize("path, data", [
    ("api/structure/export", {"source": {"path": "structures/MgO.cif"}, "format": "cif"}),
    ("api/held/release", {"ids": ["x"]}),
    ("api/held/claim", {"ids": ["x"]}),
])
def test_file_routes_require_token_host_and_xrw(served, path, data):
    base = served[0]
    assert U.post(base, path, data, token=False)[0] == 403
    port = base.split("/")[2].rsplit(":", 1)[1]
    assert U.post(base, path, data, headers={"Host": f"evil.example:{port}"})[0] == 421
    assert U.post(base, path, data, headers={"X-Requested-With": "x"})[0] == 403
    assert U.req(base, "api/held?id=x", token=False)[0] == 403
