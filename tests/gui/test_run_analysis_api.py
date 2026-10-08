"""The run-analysis routes (PLAN_analysis C2, C6, C7): /api/out/convergence,
/api/out/problems, /api/out/meta and /api/out/track on the real `mace gui`
subprocess, real outputs copied from the corpus (found by name or by
property at test time; nothing of the corpus is committed)."""
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import runcorpus as RC

sys.path[:0] = [p for p in (str(H.REPO_ROOT), str(H.REPO_ROOT / "mace"), str(H.REPO_ROOT / "Crystal_d3")) if p not in sys.path]
pytest.importorskip("numpy")

LG4 = RC.CORPUS / "OPT" / "4LG_FSI_TopMiddle_2x2_ABAB_opt.out"
SUPERCELL = RC.CORPUS / "FREQ" / "1_dia_opt_rev1_freq_B3LYP-D3-D3_optimized_supercel222.out"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    RC.need(RC.DIA_OPT)
    tmp = tmp_path_factory.mktemp("runapi")
    root = H.examples_copy(tmp)
    runs = root / "runs"
    RC.copy_into(runs, RC.DIA_OPT, RC.DIA_OPT.with_suffix(".d12"), LG4, SUPERCELL)
    first = next(p for p in sorted((RC.CORPUS / "OPT").glob("*.out")) if "SATISFIED AFTER THE FIRST" in p.read_text(errors="replace"))
    shutil.copy2(first, runs / "first_cycle.out")
    for p in sorted((RC.CORPUS / "FAILED_QA").glob("*.out")):
        RC.copy_into(root / "failed", p)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield base, root
    proc.terminate()
    proc.wait(timeout=15)


def _get(base, route, **q):
    import urllib.parse
    st, h, r = U.req(base, f"api/{route}?" + urllib.parse.urlencode(q))
    return st, h, r


def _result(base, st, r, timeout=120):
    if st == 202:
        tid = r["task"]["id"]
        end = time.time() + timeout
        while time.time() < end:
            st, _, r = U.req(base, f"api/tasks/{tid}/result")
            if st == 200:
                return r
            time.sleep(0.1)
        raise AssertionError("task did not finish")
    assert st == 200, r
    return r


# ------------------------------------------------------------------ C2-a
def test_convergence_route_equals_the_parser_and_answers_304(served):
    base, root = served
    st, h, r = _get(base, "out/convergence", path="runs/1_dia_opt_rev1.out")
    assert st == 200
    from mace.utils import out_convergence as oc
    ref = oc.parse(root / "runs" / "1_dia_opt_rev1.out")
    assert [p["scf_cycles"] for p in r["points"]] == [6, 5, 5, 3]
    assert [p["frame"] for p in r["points"]] == [0, 1, 2, 3] and r["final_frame"] == 4
    assert r["points"] == json.loads(json.dumps(ref["points"]))
    assert r["opt"]["end"]["reason"] == "all_criteria" and r["command"].startswith("mace analyze --convergence")
    st2, _, _ = U.req(base, "api/out/convergence?path=runs/1_dia_opt_rev1.out", headers={"If-None-Match": h["ETag"]})
    assert st2 == 304


# ------------------------------------------------------------------ C2-b
def test_failed_runs_stop_row_is_the_checkers(served):
    base, root = served
    from mace.completion_checker import ERROR_DESCRIPTIONS, categorize_output_file
    import contextlib, io
    for p in sorted((root / "failed").glob("*.out")):
        st, _, r = _get(base, "out/problems", path=f"failed/{p.name}")
        assert st == 200, r
        with contextlib.redirect_stdout(io.StringIO()):
            cat, _ = categorize_output_file(p)
        stop = [x for x in r["problems"] if x["code"] == "stop"][0]
        if cat in ERROR_DESCRIPTIONS:
            assert stop["message"] == ERROR_DESCRIPTIONS[cat] and stop["severity"] == "error", (p.name, stop)
        lines = p.read_text(errors="replace").splitlines()
        for row in r["problems"]:
            if row.get("line"):
                assert 0 < row["line"] <= len(lines)


# ------------------------------------------------------------------ C2-c, C2-d
def test_converged_4lg_and_first_cycle_stop(served):
    base, root = served
    st, _, r = _get(base, "out/problems", path="runs/4LG_FSI_TopMiddle_2x2_ABAB_opt.out")
    rows = r["problems"]
    fin = [x for x in rows if x["code"] == "finalrun"]
    assert len(fin) == 1 and fin[0]["severity"] == "info" and len(fin[0]["lines"]) == 3, fin
    stop = [x for x in rows if x["code"] == "stop"][0]
    assert stop["message"].startswith("Converged") and stop["severity"] == "info"
    assert not [x for x in rows if x["severity"] in ("warning", "error")], rows
    st, _, r = _get(base, "out/problems", path="runs/first_cycle.out")
    stop = [x for x in r["problems"] if x["code"] == "stop"][0]
    assert stop["severity"] == "warning" and "Stopped after the first point" in stop["message"]


# ------------------------------------------------------------------ C2-e
def test_meta_card_equals_the_extractor_and_creates_no_db(served):
    base, root = served
    before = sorted(p.name for p in (root / "runs").iterdir())
    st, _, r = _get(base, "out/meta", path="runs/1_dia_opt_rev1.out")
    r = _result(base, st, r)
    m = r["meta"]
    from mace.utils.property_extractor import CrystalPropertyExtractor
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    text = (root / "runs" / "1_dia_opt_rev1.out").read_text()
    assert m["energy_au"] == x._extract_energy_properties(text)["total_energy_au"]
    assert m["cpu_s"] == x._extract_computational_properties(text)["total_cpu_time"]
    assert m["calc_type"] == "OPT" and m["atoms"] == 2 and m["scf_cycles"] == {"min": 3, "median": 5.0, "max": 6}
    assert sorted(p.name for p in (root / "runs").iterdir()) == before
    assert not list(root.rglob("*.db")) or all(p.name == "materials.db" for p in root.rglob("*.db"))


@pytest.mark.skipif(os.environ.get("MACE_GUI_BENCH") != "1", reason="bench: MACE_GUI_BENCH=1")
def test_meta_cold_and_warm_and_the_server_stays_responsive(served):
    base, root = served
    big = max([p for p in RC.corpus_outs("FREQ", "SP")], key=lambda p: p.stat().st_size)
    RC.copy_into(root / "bench", big)
    rel = f"bench/{big.name}"
    lat = []
    stop = threading.Event()

    def ping():
        while not stop.is_set():
            t0 = time.perf_counter()
            U.req(base, "api/info")
            lat.append(time.perf_counter() - t0)
            time.sleep(0.05)
    th = threading.Thread(target=ping)
    th.start()
    t0 = time.perf_counter()
    st, _, r = _get(base, "out/meta", path=rel, wait="0")
    _result(base, st, r)
    cold = time.perf_counter() - t0
    stop.set()
    th.join()
    t0 = time.perf_counter()
    st, _, r = _get(base, "out/meta", path=rel)
    warm = time.perf_counter() - t0
    assert st == 200 and cold < 2.0 and warm < 0.05, (cold, warm)
    assert max(lat) < 0.1, max(lat)


# ------------------------------------------------------------------ C2-f
def test_imaginary_modes_noise_and_warning(served):
    base, root = served
    st, _, r = _get(base, "out/problems", path=f"runs/{SUPERCELL.name}")
    assert not [x for x in r["problems"] if x["code"] == "imaginary"]
    from mace.gui import runinfo
    assert runinfo.imaginary([0, 0, 0, -5, 300], False, False, 20)["severity"] == "info"
    assert runinfo.imaginary([0, 0, 0, -80, 300], False, False, 20)["severity"] == "warning"
    assert runinfo.imaginary([0, 0, 0, 1, 300], False, False, 20) is None


def test_imaginary_modes_agree_with_an_oracle_on_the_freq_corpus(tmp_path):
    """The route's rule and an independent oracle (fact 6: drop the 3 or 6
    modes nearest zero, then |nu| < 20 is noise) agree file by file."""
    import re
    from mace.gui import runinfo
    from mace.utils import out_convergence as oc
    outs = RC.corpus_outs("FREQ")
    tally = {"none": 0, "info": 0, "warning": 0}
    for src in outs:
        text = src.read_text(errors="replace")
        rows = re.findall(r"^\s*(\d+)-\s*(\d+)\s+[-\d.E+]+\s+([-\d.E+]+)\s+[-\d.E+]+\s+\(", text, re.M)
        freqs = []
        seen = set()
        for a, b, f in rows:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            freqs += [float(f)] * (int(b) - int(a) + 1)
        if not freqs:
            continue
        mol = "DIMENSIONALITY OF THE SYSTEM      0" in text
        drop = 6 if mol else 3
        rest = sorted(freqs, key=abs)[drop:]
        neg = [f for f in rest if f < 0]
        want = "none" if not neg else ("warning" if any(abs(f) >= 20 for f in neg) else "info")
        (p,) = RC.copy_into(tmp_path, src)
        conv = oc.parse(p)
        got = [x for x in runinfo.problems(p, {**conv, "live": False}, 20.0) if x["code"] == "imaginary"]
        g = got[0]["severity"] if got else "none"
        if not (mol and runinfo._linear_molecule(text)):
            assert g == want, (src.name, g, want)
        tally[g] += 1
        p.unlink()
    assert sum(tally.values()) > 50, tally


# ------------------------------------------------------------------ C7-c (unit, every corpus output)
def test_problems_never_raise_and_completed_runs_have_no_errors(tmp_path):
    from mace.gui import runinfo
    from mace.utils import out_convergence as oc
    import contextlib, io
    from mace.completion_checker import categorize_output_file
    outs = [p for p in RC.corpus_outs("OPT", "SP", "FAILED_QA") if p.stat().st_size < 5_000_000]
    for src in outs:
        (p,) = RC.copy_into(tmp_path, src)
        rows = runinfo.problems(p, {**oc.parse(p), "live": False}, 20.0)
        with contextlib.redirect_stdout(io.StringIO()):
            cat, _ = categorize_output_file(p)
        if cat in ("complete", "completesp"):
            assert not [x for x in rows if x["severity"] == "error"], (src.name, rows)
        p.unlink()
    (p,) = RC.copy_into(tmp_path, LG4)
    assert not [x for x in runinfo.problems(p, {**oc.parse(p), "live": False}) if x["severity"] in ("warning", "error")]


# ------------------------------------------------------------------ C2-g, C6-c (tracking across a cell face)
def _synth_frames(fracs, a=3.0):
    import numpy as np
    cell = np.eye(3) * a
    return [{"cell": cell, "frac": np.array([[0.9, 0.5, 0.5], [f, 0.5, 0.5]]), "pbc": [True, True, True]} for f in fracs]


def test_track_is_continuous_across_the_cell_face():
    from mace.gui import runinfo
    frames = _synth_frames([0.97, 0.99, 0.01, 0.03])
    m = {"type": "distance", "atoms": [{"base": 0, "shift": [0, 0, 0]}, {"base": 1, "shift": [0, 0, 0]}]}
    vals = runinfo.track(frames, m)
    steps = [abs(b - a) for a, b in zip(vals, vals[1:])]
    assert max(steps) < 0.07, vals                     # 0.02 * 3 Å per frame, never a lattice vector
    import numpy as np
    raw = [abs((f["frac"][1] - f["frac"][0])[0]) * 3.0 for f in frames]
    assert max(abs(b - a) for a, b in zip(raw, raw[1:])) > 2.0     # the wrapped value would jump


def test_track_refuses_bad_atoms():
    from mace.gui import runinfo
    frames = _synth_frames([0.97, 0.99])
    for atoms in ([{"base": -1}, {"base": 0}], [{"base": 0}, {"base": 5}], [{"base": 0}, {"base": 1, "shift": [1, 2]}],
                  [{"base": "x"}, {"base": 1}], [{"shift": [0, 0, 0]}, {"base": 1}]):
        with pytest.raises(ValueError):
            runinfo.track(frames, {"type": "distance", "atoms": atoms})


def test_track_on_a_real_run_equals_ase(served):
    base, root = served
    ase_geom = pytest.importorskip("ase.geometry")
    import numpy as np
    from mace.gui import outindex
    idx = outindex.build(root / "runs" / "1_dia_opt_rev1.out")
    m = {"type": "distance", "atoms": [{"base": 0, "shift": [0, 0, 0]}, {"base": 1, "shift": [0, 0, 0]}]}
    st, _, r = U.post(base, "api/out/track", {"path": "runs/1_dia_opt_rev1.out", "measurement": m})
    assert st == 200 and len(r["values"]) == idx.n_steps
    for k in (0, idx.n_steps - 1):
        s = idx.block_structure(k)
        d = np.asarray(s["cart"])[1] - np.asarray(s["cart"])[0]
        assert abs(r["values"][k] - float(np.linalg.norm(d))) < 1e-6
