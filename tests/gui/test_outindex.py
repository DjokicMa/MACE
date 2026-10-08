"""The byte-offset index of CRYSTAL outputs and the explorer's file details
(plan T11, B6): the index gives the same frames as read_out_frames (count,
labels, energies, coordinates to 1e-12) on every demo output and on a
generated 34 MB run; /api/out/summary and /api/out/step stay small and fast;
the Trajectory document asks for the summary and one step per scrub; file
details are read in the pool and cached.

The 34 MB file is made from examples/gui/outputs/Ag2Br3_opt_unfinished.out
(shipped with the repo) by repeating one of its OPT-point blocks; it lives in
tmp and is deleted at teardown. No corpus or research file is read."""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

np = pytest.importorskip("numpy")
pytest.importorskip("ase")

DEMO_OUT = H.EXAMPLES / "outputs" / "Ag2Br3_opt_unfinished.out"
BIG_MB = 34


def make_big(path: Path, mb: int = BIG_MB) -> int:
    """The demo OPT output with its point-4 block repeated until the file
    reaches `mb` MB (the header once, the remaining points at the end)."""
    src = DEMO_OUT.read_text().split("\n")
    head, rep, tail = src[:384], src[384:457], src[457:]
    assert rep[0].strip().startswith("COORDINATE AND CELL OPTIMIZATION - POINT"), rep[0]
    blk = "\n".join(rep) + "\n"
    n = (mb * 2**20 - len("\n".join(head + tail))) // len(blk) + 1
    with open(path, "w") as fh:
        fh.write("\n".join(head) + "\n")
        for _ in range(n):
            fh.write(blk)
        fh.write("\n".join(tail))
    return n


def room_for_big():
    if not U.tmp_free_ok(2):
        return "under 2 GB free in /tmp"
    if not U.mem_ok(4):
        return "under 4 GB of memory available"
    return ""


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    why = room_for_big()
    if why:
        pytest.skip(f"the 34 MB output fixture needs room: {why}")
    d = tmp_path_factory.mktemp("bigout")
    p = d / "big_matdump.out"
    make_big(p)
    yield p
    p.unlink(missing_ok=True)


def _mods():
    sys.path[:0] = [p for p in (str(H.REPO_ROOT / "mace"), str(H.REPO_ROOT / "Crystal_d3")) if p not in sys.path]
    from mace.gui import outindex, structures
    return outindex, structures


def assert_same(path):
    outindex, S = _mods()
    ref = S.read_out_frames(Path(path))
    idx = outindex.build(Path(path))
    fr = ref["frames"]
    assert idx.n_steps == len(fr)
    assert idx.labels == [f["label"] for f in fr]
    assert idx.energies() == [f["energy_au"] for f in fr]
    assert idx.status == ref["opt_status"] and S._DIM_NAMES[idx.dim] == ref["dimensionality"]
    for k in range(len(fr)):
        a, b = fr[k]["structure"], idx.block_structure(k)
        assert a["numbers"] == b["numbers"] and a["pbc"] == b["pbc"]
        assert np.max(np.abs(np.asarray(a["cart"]) - np.asarray(b["cart"]))) <= 1e-12
        assert np.max(np.abs(np.asarray(a["cell"]) - np.asarray(b["cell"]))) <= 1e-12
    return idx


@pytest.mark.parametrize("name", sorted(p.name for p in (H.EXAMPLES / "outputs").glob("*.out")))
def test_outindex_equals_read_out_frames(name):
    assert_same(H.EXAMPLES / "outputs" / name)


def test_outindex_equals_read_out_frames_big(big):
    idx = assert_same(big)
    assert idx.n_steps > 8000


@pytest.fixture(scope="module")
def served(tmp_path_factory, big):
    tmp = tmp_path_factory.mktemp("outsrv")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    shutil.copy(big, root / "outputs" / "big_matdump.out")
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield base, root
    proc.terminate()
    proc.wait(timeout=15)
    (root / "outputs" / "big_matdump.out").unlink(missing_ok=True)


def test_out_summary_bytes_and_time(served):
    base = served[0]
    t0 = time.perf_counter()
    st, h, r = U.req(base, "api/out/summary?path=outputs/big_matdump.out")
    cold = time.perf_counter() - t0
    assert st == 200, r
    raw = len(json.dumps(r))
    assert raw <= 300 * 1024, raw
    assert cold <= 2.5, cold
    t0 = time.perf_counter()
    st, h2, r2 = U.req(base, "api/out/summary?path=outputs/big_matdump.out")
    hit = time.perf_counter() - t0
    assert st == 200 and "cache;desc=hit" in h2.get("Server-Timing", "")
    assert hit <= 0.05, hit
    assert r["n_steps"] == sum(c for _, c in r["label_runs"]) and len(r["energies_au"]) == r["n_steps"]
    assert set(r["frames_at"]) == {"0", str(r["n_steps"] - 1)} and r["comparison"]["max_displacement"] >= 0
    # a 304 revalidation
    assert U.req(base, "api/out/summary?path=outputs/big_matdump.out", headers={"If-None-Match": h2["ETag"]})[0] == 304
    # the small demo run: the same answer as /api/trajectory, in the compact shape
    _, _, s = U.req(base, "api/out/summary?path=outputs/Ag2Br3_opt_unfinished.out")
    _, _, t = U.req(base, "api/trajectory?path=outputs/Ag2Br3_opt_unfinished.out")
    labels = [lab for lab, c in s["label_runs"] for _ in range(c)]
    assert labels == t["labels"] and s["energies_au"] == t["energies_au"] and s["n_steps"] == len(t["frames"])
    assert s["frames_at"]["0"]["cart"] == t["frames"][0]["cart"]
    assert s["comparison"]["max_displacement"] == t["comparison"]["max_displacement"]


def test_out_step_bytes(served):
    base = served[0]
    _, _, s = U.req(base, "api/out/summary?path=outputs/big_matdump.out")
    for k in (1, s["n_steps"] // 2, s["n_steps"] - 2):
        st, _, r = U.req(base, f"api/out/step?path=outputs/big_matdump.out&step={k}")
        assert st == 200 and r["step"] == k, r
        assert len(json.dumps(r)) <= 40 * 1024
        assert r["energy_au"] == s["energies_au"][k]
    assert U.req(base, f"api/out/step?path=outputs/big_matdump.out&step={s['n_steps']}")[0] == 400
    # /api/structure of an .out reads only that frame, through the index
    _, _, a = U.req(base, "api/out/step?path=outputs/Ag2Br3_opt_unfinished.out&step=2")
    _, _, b = U.req(base, "api/structure?path=outputs/Ag2Br3_opt_unfinished.out&frame=2")
    assert a["cart"] == b["cart"] and a["energy_au"] == b["energy_au"]


def test_files_details_cache_hit(served):
    base = served[0]
    st, _, r = U.req(base, "api/files?details=1")
    d = r["details"]
    assert d["structures/MgO.cif"]["formula"] == "MgO" and d["structures/MgO.cif"]["n"] == 8
    assert d["structures/MgO.cif"]["sg_number"] == 225
    assert d["outputs/Ag2Br3_opt_unfinished.out"]["n"] == 10
    assert all("error" not in v for k, v in d.items() if not k.endswith(".d12")), d
    t0 = time.perf_counter()
    st, _, r2 = U.req(base, "api/files?details=1")
    assert r2["details"] == d
    assert time.perf_counter() - t0 < 1.0
    rep = U.req(base, "api/test/held")[2]["parse"]
    assert rep["hits"] >= len(d)
    # a small folder's batch is quiet: no task in the list (or in the page's status bar)
    assert not [t for t in U.req(base, "api/tasks")[2]["tasks"] if t["kind"] == "details"]
