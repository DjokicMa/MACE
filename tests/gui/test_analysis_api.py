"""Analysis › Structure Analysis through the real `mace gui` (POST
/api/analysis/run as a background task, GET /api/analysis/info,
POST /api/analysis/rcsr/import), on the bundled example structures: the same
numbers the mace.analysis functions give, every result JSON-clean."""
import json
import math
import time

import pytest

import guiharness as H
import p4util as U

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("spglib")
pytest.importorskip("ase")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("analysis")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2",
                                   MACE_CONFIG_DIR=str(tmp / "cfg"))
    yield base, root
    proc.terminate()
    proc.wait(timeout=15)


def run(base, tool, path=None, params=None, timeout=120, **extra):
    body = {"tool": tool, "params": params or {}, "wait": 5, **extra}
    if path:
        body["source"] = {"path": path, "frame": -1}
    st, _, r = U.post(base, "api/analysis/run", body)
    if st == 202:
        tid = r["task"]["id"]
        end = time.time() + timeout
        while time.time() < end:
            st, _, r = U.req(base, f"api/tasks/{tid}/result")
            if st == 200:
                break
            time.sleep(0.2)
    assert st == 200, (st, r)
    json.dumps(r, allow_nan=False)              # strict JSON: no NaN or Infinity leaks out
    return r["result"]


def test_info_lists_probes_sources_and_status(served):
    base, _ = served
    st, _, r = U.req(base, "api/analysis/info")
    assert st == 200
    assert r["probes"]["N2"]["radius"] == pytest.approx(1.82) and "CuKa" in r["sources"]
    assert "dia" in r["rcsr"]["builtin"] and r["rcsr"]["imported"] == 0
    assert "porosity" in r["tools"] and "topology" in r["tools"]


def test_porosity_of_a_dense_crystal_and_render_data(served):
    base, _ = served
    r = run(base, "porosity", "structures/MgO.cif", {"probe": "He", "spacing": 0.25})
    assert r["AV"]["fraction"] == 0 and r["probe"]["name"] == "He"
    assert r["render"]["shape"][0] >= 5 and not r["render"]["has_accessible"]
    r2 = run(base, "porosity", "structures/MgO.cif", {"probe_radius": 0.3, "spacing": 0.25})
    assert r2["probe"]["radius"] == pytest.approx(0.3) and r2["POAV"]["fraction"] == 0   # hard spheres fill rock salt


def test_pxrd_of_silicon(served):
    base, _ = served
    r = run(base, "pxrd", "structures/Si.cif", {"source": "CuKa", "min": 20, "max": 80})
    first = r["peaks"][0]
    assert first["two_theta"] == pytest.approx(28.442, abs=0.01) and first["hkl"] == [1, 1, 1]
    assert len(r["two_theta"]) == len(r["intensity"]) and max(r["intensity"]) == pytest.approx(100)


def test_pawley_from_pattern_text(served):
    base, _ = served
    from mace.analysis import refine, structure
    si = structure.load(H.EXAMPLES / "structures" / "Si.cif")
    pat = refine.synthetic_pattern(si, zero=0.02, two_theta_range=(20, 90))
    text = "\n".join(f"{x:.4f} {y:.1f}" for x, y in zip(pat["two_theta"], pat["intensity"]))
    r = run(base, "refine", "structures/Si.cif", {"method": "pawley"}, pattern_text=text, pattern_name="si.xy")
    assert r["lattice"]["a"] == pytest.approx(5.431, abs=5e-4) and r["zero"] == pytest.approx(0.02, abs=0.006)
    pk = run(base, "peaks", None, {"prominence": 0.03}, pattern_text=text, pattern_name="si.xy")
    assert pk["peaks"][0]["two_theta"] == pytest.approx(28.46, abs=0.03)


def test_topology_polyhedra_morphology_planes(served):
    base, _ = served
    t = run(base, "topology", "structures/diamond.cif")
    assert t["summary"]["topologies"] == ["dia"] and t["net_edges"]
    p = run(base, "polyhedra", "structures/MgO.cif")
    assert p["polyhedra"] and p["polyhedra"][0]["cn"] == 6 and p["polyhedra"][0]["shape"] == "octahedron"
    m = run(base, "morphology", "structures/NaCl.cif", {"energies": "100:1.0, 111:1.5"})
    assert m["families"][0]["hkl"] == [1, 0, 0] and m["triangles"]
    pl = run(base, "planes", "structures/Si.cif", {"hkl": "1 1 1", "offset": 0.5})
    assert pl["d"] == pytest.approx(5.431 / math.sqrt(3), abs=1e-3) and pl["planes"]
    c = run(base, "cleavage", "structures/Si.cif")
    assert c["families"][0]["hkl"] == [1, 1, 1]
    b = run(base, "bonds", "structures/Si.cif")
    assert list(b["bonds"]) == ["Si-Si"] and b["bonds"]["Si-Si"]["mean"] == pytest.approx(2.3517, abs=1e-3)


def test_bad_requests(served):
    base, _ = served
    st, _, r = U.post(base, "api/analysis/run", {"tool": "nope"})
    assert st == 400 and "tool" in r["error"]
    st, _, r = U.post(base, "api/analysis/run", {"tool": "peaks", "params": {}})
    assert st == 400 and "pattern" in r["error"]


def test_rcsr_import_from_the_folder(served):
    base, root = served
    cs = [6, 18, 38, 66, 102, 146, 198, 258, 326, 402]
    blk = "\n".join(["start", "1", "zzz", "1a", " 0  !n", " 0  !n", " 0  !n", " 0  !n", " 0  !n", "Pm-3m 221", " 1 1 1 90 90 90",
                     "1", "V1 6", " 0 0 0", " 0,0,0", " 1 a", " m-3m", " 48", "1", "E1 2", " .5 0 0", " 1/2,0,0", " 3 d", " 4/mmm",
                     "1", "1", "1", "[x]", "unk", " " + " ".join(map(str, cs)) + " 1561", " x", "1", "1"])
    (root / "mini3dall.txt").write_text(blk + "\n")
    st, _, r = U.post(base, "api/analysis/rcsr/import", {"path": "mini3dall.txt"})
    assert st == 200 and r["imported"] == 1
    st, _, info = U.req(base, "api/analysis/info")
    assert info["rcsr"]["imported"] == 1
    t = run(base, "topology", "structures/NaCl.cif")
    assert [m["name"] for m in t["nets"][0]["matches"]][:1] == ["zzz"]       # imported nets are tried first


def test_a_batch_of_three_structures(served):
    """What Analyze Structures… sends (one run per file, the file as the source and as the task's
    document): each answer carries the row of `mace analyze <tool> FILE... --csv`, the task is named
    by tool and structure, and runs of different files never replace each other."""
    base, _ = served
    from mace.analysis.cli import summary_row
    files = ["structures/Si.cif", "structures/MgO.cif", "structures/diamond.cif"]
    params = {"probe": "He", "spacing": 0.3, "render": False}
    tids = []
    for f in files:
        st, _, r = U.post(base, "api/analysis/run", {"tool": "porosity", "params": params, "source": {"path": f, "frame": -1},
                                                       "doc": f"file:{f}", "wait": 0})
        assert st in (200, 202), (st, r)
        tids.append(r["task"]["id"])
    rows = []
    for f, tid in zip(files, tids):
        end = time.time() + 120
        while time.time() < end:
            st, _, r = U.req(base, f"api/tasks/{tid}/result")
            if st == 200:
                break
            time.sleep(0.2)
        assert st == 200, (f, st, r)
        assert r["task"]["title"] == f"Porosity: {f.split('/')[-1][:-4]}" and r["task"]["state"] == "done"
        assert "render" not in r["result"] and r["summary"] == json.loads(json.dumps(summary_row("porosity", r["result"], f, params)))
        rows.append(r["summary"])
    assert [x["file"] for x in rows] == files and all(x["probe"] == "He" and x["spacing"] == pytest.approx(0.3, abs=0.02) for x in rows)
