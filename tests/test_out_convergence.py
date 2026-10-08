"""The shared convergence parser mace/utils/out_convergence.py and
`mace analyze --convergence` (PLAN_analysis C1).

Real outputs are copied from the gitignored test/ corpus into tmp (skipped
without it); the stop cases the corpus lacks use the generic synthetic
outputs of tests/synth_out.py."""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "test"
sys.path[:0] = [p for p in (str(REPO), str(REPO / "mace"), str(REPO / "Crystal_d3"), str(Path(__file__).parent))
                if p not in sys.path]

from mace.utils import out_convergence as oc  # noqa: E402
import synth_out as SO  # noqa: E402

DIA = CORPUS / "OPT" / "1_dia_opt_rev1.out"
LG4 = CORPUS / "OPT" / "4LG_FSI_TopMiddle_2x2_ABAB_opt.out"


def _copy(tmp_path, src):
    if not src.is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    dst = tmp_path / src.name
    shutil.copyfile(src, dst)
    return dst


def _env():
    return {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}


# ------------------------------------------------------------------ C1-a
def test_dia_points_segments_end_and_scf(tmp_path):
    r = oc.parse(_copy(tmp_path, DIA))
    pts = r["points"]
    assert [p["point"] for p in pts] == [1, 2, 3, 4]
    assert [p["scf_cycles"] for p in pts] == [6, 5, 5, 3]
    assert [p["de_printed"] for p in pts] == [None, -4.549e-5, -1.012e-5, 2.783e-5]
    p1 = pts[0]
    assert p1["max_grad"] == 0.021242 and p1["thr"]["max_grad"] == 0.00045 and p1["max_disp"] is None
    assert r["thresholds"]["maxgrad"]["source"] == "out"
    assert all(pts[2]["converged"].values())
    assert r["segments"] == [{"index": 0, "first_point": 1, "last_point": 3, "restart_line": 805},
                             {"index": 1, "first_point": 4, "last_point": 4, "restart_line": None}]
    end = r["opt"]["end"]
    assert end["reason"] == "all_criteria" and end["points"] == 4 and end["energy"] == -76.21718773781
    det = r["scf"]["detail"]
    assert len(det) == 1 and det[0]["point"] == 1 and [c["cyc"] for c in det[0]["cycles"]] == list(range(7))
    assert det[0]["cycles"][-1]["detot"] == 8.87e-09 and det[0]["ended"] == "energy"
    assert r["scf"]["moved_to_scfout_log"] is True
    assert r["thresholds"]["toldee_scf"]["value"] == 1e-7 and r["thresholds"]["toldee_scf"]["source"] == "out"
    assert r["gradient_units"] == "internal"
    assert abs(pts[0]["energy_au"] - -76.217159957139) < 1e-10          # the corrected first point (C0)


# ------------------------------------------------------------------ C1-b, C1-f
def test_frame_mapping_dia(tmp_path):
    r = oc.parse(_copy(tmp_path, DIA))
    assert oc.frame_of_point(r["points"], [b for b in _frame_lines(tmp_path / DIA.name)]) == {1: 0, 2: 1, 3: 2, 4: 3}
    assert r["final_frame"] == 4 and r["n_frames"] == 5


def _frame_lines(path):
    from mace.gui import outindex
    return [b.line + 1 for b in outindex.build(path).blocks]


def test_frames_energies_and_cells_match_outindex_on_every_real_opt(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("ase")
    from ase.geometry import cell_to_cellpar
    from mace.gui import outindex
    outs = sorted((CORPUS / "OPT").glob("*.out")) if CORPUS.is_dir() else []
    if not outs:
        pytest.skip("test/ corpus not present (gitignored)")
    checked = 0
    for src in outs:
        p = _copy(tmp_path, src)
        idx = outindex.build(p)
        flines = [b.line + 1 for b in idx.blocks]
        if idx.molecule:
            # a molecule's frames are its Angstrom tables (outindex, Area E), which the parser's
            # own detection (PRIMITIVE CELL blocks) does not look for; Run Analysis gives each
            # table to the first point after it only (runinfo.convergence)
            from mace.gui import runinfo
            r = runinfo.convergence(p, idx)
            framed = [q for q in r["points"] if q["frame"] is not None]
            assert len({q["frame"] for q in framed}) == len(framed) <= idx.n_steps, src.name
            if r["points"]:
                assert r["points"][0]["frame"] == 0, src.name
            p.unlink()
            continue
        r = oc.parse(p)
        # the parser's own block detection finds outindex's frames
        assert r["n_frames"] == len(flines), src.name
        r2 = oc.parse(p, frame_lines=flines)
        assert [q["frame"] for q in r2["points"]] == [q["frame"] for q in r["points"]], src.name
        final = 1 if "final" in idx.labels else 0
        if idx.n_steps:
            pts_with_frames = [q for q in r["points"] if q["frame"] is not None]
            assert len({q["frame"] for q in pts_with_frames}) + final <= idx.n_steps
            oe = idx.opt_energies()
            for q in pts_with_frames:
                assert q["energy_au"] == oe[q["frame"]], (src.name, q["point"])
                s = idx.block_structure(q["frame"])
                par = cell_to_cellpar(np.asarray(s["cell"]))
                dim = idx.dim
                c = q["cell"]
                for i, key in enumerate(("a", "b", "c")[:dim]):
                    assert abs(par[i] - c[key]) < 1e-6, (src.name, key)
                if dim == 3:
                    assert abs(abs(np.linalg.det(np.asarray(s["cell"]))) - c["volume"]) < 1e-3 * c["volume"]
            if r["final_frame"] is not None:
                assert idx.labels[r["final_frame"]] == "final"
            checked += 1
        p.unlink()
    assert checked >= 40


# ------------------------------------------------------------------ C1-c
def test_every_real_output_parses(tmp_path):
    outs = []
    for sub in ("OPT", "SP", "FREQ", "FAILED_QA"):
        outs += sorted((CORPUS / sub).glob("*.out")) if CORPUS.is_dir() else []
    if not outs:
        pytest.skip("test/ corpus not present (gitignored)")
    first_cycle = 0
    for src in outs:
        p = _copy(tmp_path, src)
        r = oc.parse(p)
        json.dumps(r)
        oc.render_text(r)
        has = "SATISFIED AFTER THE FIRST" in p.read_text(errors="replace")
        if has:
            assert r["opt"]["end"]["reason"] == "gradient_first_cycle", src.name
            first_cycle += 1
        p.unlink()
    assert first_cycle >= 10
    r = oc.parse(_copy(tmp_path, LG4))
    assert r["opt"]["end"]["reason"] == "all_criteria" and len(r["segments"]) == 4


# ------------------------------------------------------------------ C1-d (synthetic stops)
def _w(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


Y4, N4 = (True,) * 4, (False,) * 4


def test_gradient_and_energy_stop(tmp_path):
    t = SO.opt_output([-76.0, -76.00001, -76.0000100500], [None, -1e-5, -5e-8],
                      [((0.01, 0.01, 0, 0), N4), ((0.001, 0.001, 0.01, 0.01), N4),
                       ((0.0001, 0.0001, 0.002, 0.0015), (True, True, False, False))],
                      warning="GEOMETRY OPTIMIZATION STOPPED")
    r = oc.parse(_w(tmp_path, "ge.out", t))
    assert r["opt"]["end"]["reason"] == "gradient_and_energy"
    assert "GEOMETRY OPTIMIZATION STOPPED" in r["opt"]["end"]["warning_lines"]
    assert "manual p.208" in oc.stop_sentence(r)


def test_energy_stalled_stop(tmp_path):
    es = [-76.0, -76.00000001, -76.00000002, -76.00000003, -76.00000004]
    t = SO.opt_output(es, [None, -1e-8, -1e-8, -1e-8, -1e-8], [((0.01, 0.01, 0, 0), N4)] + [((0.01, 0.01, 0.01, 0.01), N4)] * 4)
    r = oc.parse(_w(tmp_path, "es.out", t))
    assert r["opt"]["end"]["reason"] == "energy_stalled"


def test_crystal23_stop_texts_decide_only_what_the_state_leaves_open(tmp_path):
    """The sentences read from the CRYSTAL23 binary are extra evidence: a run whose state is inconclusive
    (DE not printed below TOLDEE) but whose banner says the surface is flat is energy_stalled; the same
    banner never overrides a state that decides (all four criteria met)."""
    es = [-76.0, -76.0001, -76.0002]
    crits = [((0.01, 0.01, 0, 0), N4)] + [((0.01, 0.01, 0.01, 0.01), N4)] * 2
    t = SO.opt_output(es, [None, -1e-4, -1e-4], crits, warning="FLAT SURFACE - NO CHANGE IN ENERGY FOR    4 STEPS")
    r = oc.parse(_w(tmp_path, "flat.out", t))
    assert r["opt"]["end"]["reason"] == "energy_stalled"
    t = SO.opt_output(es, [None, -1e-4, -1e-4], crits[:2] + [((0.0001, 0.0001, 0.0001, 0.0001), Y4)],
                      warning="FLAT SURFACE - NO CHANGE IN ENERGY FOR    4 STEPS")
    assert oc.parse(_w(tmp_path, "flat2.out", t))["opt"]["end"]["reason"] == "all_criteria"
    t = SO.opt_output(es, [None, -1e-4, -1e-4], crits, warning="CONVERGENCE ON GRADIENT - NO CHANGE IN ENERGY FOR    4 STEPS")
    assert oc.parse(_w(tmp_path, "cg.out", t))["opt"]["end"]["reason"] == "gradient_and_energy"


def test_maxcycle_and_failed(tmp_path):
    es = [-76.0, -76.001, -76.002]
    crits = [((0.01, 0.01, 0, 0), N4)] + [((0.01, 0.01, 0.01, 0.01), N4)] * 2
    r = oc.parse(_w(tmp_path, "mc.out", SO.opt_output(es, [None, -1e-3, -1e-3], crits, end=None, maxcycle=3)))
    assert r["opt"]["end"]["reason"] == "maxcycle"
    r = oc.parse(_w(tmp_path, "un.out", SO.opt_output(es, [None, -1e-3, -1e-3], crits, end=None, maxcycle=800)))
    assert r["opt"]["end"]["reason"] == "unfinished"
    r = oc.parse(_w(tmp_path, "fa.out", SO.opt_output(es, [None, -1e-3, -1e-3], crits, end="FAILED")))
    assert r["opt"]["end"]["reason"] == "failed"


def test_atom_only_units(tmp_path):
    t = SO.opt_output([-76.0, -76.001], [None, -1e-3], [((0.01, 0.01, 0, 0), N4), ((0.0001, 0.0001, 0.0001, 0.0001), Y4)],
                      units=False, control="ATOM")
    r = oc.parse(_w(tmp_path, "ao.out", t))
    assert r["gradient_units"] == "Ha/bohr" and r["displacement_units"] == "bohr"
    assert r["opt"]["end"]["reason"] == "all_criteria"


# ------------------------------------------------------------------ C1-e (threshold sources)
D12 = """mat1 synthetic
CRYSTAL
0 0 0
227
3.56
1
6 0.125 0.125 0.125
OPTGEOM
TOLDEG
0.0001
ENDOPT
END
"""


def _bare_out():
    t = SO.opt_output([-76.0, -76.001], [None, -1e-3], [((0.01, 0.01, 0, 0), N4), ((0.0001, 0.00005, 0.0001, 0.0001), Y4)])
    # no header table, no per-point thresholds that name TOLDEG's value: only the d12 can say
    keep = [ln for ln in t.splitlines() if "COMPONENT" not in ln and "THRESHOLD ON ENERGY" not in ln
            and "ALLOWED NUMBER" not in ln and "GRADIENT " not in ln and "DISPLAC." not in ln]
    return "\n".join(keep) + "\n"


def test_threshold_sources(tmp_path):
    out = _w(tmp_path, "mat1.out", _bare_out())
    (tmp_path / "mat1.d12").write_text(D12)
    r = oc.parse(out)
    assert r["thresholds"]["toldeg"] == {"value": 0.0001, "source": "d12"}
    (tmp_path / "mat1.d12").unlink()
    r = oc.parse(out)
    t = r["thresholds"]["toldeg"]
    assert t["source"] == "default" and t["value"] == 0.0003 and t["manual_page"] == 189
    (tmp_path / "mat1.d12").write_text(D12.replace("OPTGEOM\nTOLDEG\n0.0001\nENDOPT", "FREQCALC\nPREOPTGEOM\nENDOPT\nEND"))
    r = oc.parse(out)
    assert r["thresholds"]["toldeg"]["value"] == 0.00003 and r["thresholds"]["toldeg"]["source"] == "default"


# ------------------------------------------------------------------ C1-g (CLI = parse())
def test_cli_json_equals_parse_and_text(tmp_path):
    p = _copy(tmp_path, DIA)
    cli = [sys.executable, str(REPO / "mace_cli"), "--no-banner", "analyze", "--convergence", str(p)]
    r = subprocess.run(cli + ["--json"], capture_output=True, text=True, env=_env(), timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == json.loads(json.dumps(oc.parse(p)))
    r = subprocess.run(cli, capture_output=True, text=True, env=_env(), timeout=120)
    assert r.returncode == 0, r.stderr
    assert "Converged: all four criteria met at point 4" in r.stdout
    assert "from the .out (line" in r.stdout and "internal optimization coordinates" in r.stdout
    h = subprocess.run([sys.executable, str(REPO / "mace_cli"), "analyze", "--help"], capture_output=True, text=True,
                       env=_env(), timeout=120)
    assert "--convergence" in h.stdout


# ------------------------------------------------------------------ C1-h (bench)
@pytest.mark.skipif(os.environ.get("MACE_GUI_BENCH") != "1", reason="bench: MACE_GUI_BENCH=1")
def test_two_hundred_point_parse_under_a_second(tmp_path):
    src = (REPO / "examples" / "gui" / "outputs" / "Ag2Br3_opt_unfinished.out").read_text().split("\n")
    head, rep, tail = src[:384], src[384:457], src[457:]
    p = tmp_path / "bench200.out"
    p.write_text("\n".join(head + rep * 200 + tail))
    t0 = time.perf_counter()
    r = oc.parse(p)
    dt = time.perf_counter() - t0
    assert len(r["points"]) >= 200
    assert dt < 1.0, dt
