"""Outputs that are still being written (PLAN_analysis C18): bounded
snapshot reads that never report half a line, the incremental index, and
the replaced-file rule. A real CRYSTAL output copied from the corpus is
written into tmp in slices by this test (some ending mid-line, mid-number
and mid-table)."""
import os
import random
import shutil
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import runcorpus as RC

sys.path[:0] = [p for p in (str(H.REPO_ROOT), str(H.REPO_ROOT / "mace"), str(H.REPO_ROOT / "Crystal_d3")) if p not in sys.path]
pytest.importorskip("numpy")


def _mods():
    from mace.gui import outindex
    from mace.utils import livefile, out_convergence
    return outindex, livefile, out_convergence


def _cuts(data: bytes, n: int, seed: int = 7):
    rnd = random.Random(seed)
    cuts = sorted(rnd.sample(range(1, len(data)), n - 1)) + [len(data)]
    # make sure some cuts land mid-number and mid-table
    k = data.find(b"TOTAL ENERGY(DFT)(AU)(")
    m = data.find(b"ATOMS IN THE UNIT CELL", k)
    cuts += [k + 34, data.find(b"\n      2 ", m) + 5]           # mid-number; one atom row of two (mid-table)
    return sorted(set(c for c in cuts if 0 < c <= len(data)))


def _sig(idx):
    return ([b.line for b in idx.blocks], list(idx.labels), idx.energies(), idx.opt_energies(), idx.status,
            idx.freq_start, idx.freq_modes, idx.pending_frame)


def test_slices_never_report_half_a_line_and_extend_equals_build(tmp_path):
    outindex, livefile, oc = _mods()
    (src,) = RC.copy_into(tmp_path / "src", RC.DIA_OPT)
    data = src.read_bytes()
    final = outindex.build(src, live=False)
    final_e = [e for e in final.energies() if e is not None]
    final_lines = [b.line for b in final.blocks]
    final_points = oc.parse(src)["points"]
    live = tmp_path / "live.out"
    live.write_bytes(b"")
    prev = outindex.build(live, live=True)
    pend_seen = False
    for cut in _cuts(data, 37):
        with open(live, "wb") as fh:
            fh.write(data[:cut])
        idx = outindex.build(live, live=True)
        es = [e for e in idx.energies() if e is not None]
        assert es == final_e[:len(es)], cut                         # exact floats, never a cut number
        assert [b.line for b in idx.blocks] == final_lines[:idx.n_steps], cut
        if idx.pending_frame:
            pend_seen = True
            assert idx.n_steps < len(final_lines)
        ext = outindex.extend(prev, live, live=True)
        assert _sig(ext) == _sig(idx), cut
        prev = ext
        snap = livefile.snapshot(live)
        r = oc.parse(live, size=snap.size, complete_lines_only=True)
        got = [(p["point"], p["energy_scf_au"]) for p in r["points"]]
        want = [(p["point"], p["energy_scf_au"]) for p in final_points]
        assert got == want[:len(got)], cut
    assert pend_seen
    # finished: the whole file, as a non-live build gives it
    assert _sig(outindex.extend(prev, live, live=False))[:7] == _sig(final)[:7]


def test_truncated_file_is_replaced_and_rebuilt(tmp_path):
    outindex, livefile, _ = _mods()
    (src,) = RC.copy_into(tmp_path / "src", RC.DIA_OPT)
    data = src.read_bytes()
    live = tmp_path / "run.out"
    live.write_bytes(data[: len(data) // 2])
    a = outindex.build(live, live=True)
    s0 = livefile.snapshot(live)
    live.write_bytes(data[: len(data) // 4])              # a resubmission truncates the output
    s1 = livefile.snapshot(live)
    assert not livefile.same_file(s0, s1, live)
    b = outindex.extend(a, live, live=True)
    assert _sig(b) == _sig(outindex.build(live, live=True))


def test_small_file_growing_past_the_head_stays_the_same_file(tmp_path):
    _, livefile, _ = _mods()
    p = tmp_path / "small.out"
    p.write_bytes(b"line\n" * 1000)                       # 5 KB
    s0 = livefile.snapshot(p)
    with open(p, "ab") as fh:
        fh.write(b"more\n" * 30000)                       # past 64 KB
    s1 = livefile.snapshot(p)
    assert livefile.same_file(s0, s1, p)
    data, end = livefile.read_complete(p, livefile.snapshot(p), 0)
    assert data.endswith(b"\n") and end == len(data)


def test_finished_outputs_answer_as_before(tmp_path):
    """C18-c: every finished corpus output (up to 3 MB) gives the frames,
    labels and energies of read_out_frames, the validated reader."""
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    outindex, _, _ = _mods()
    from mace.gui import structures as S
    outs = [p for p in RC.corpus_outs("OPT", "SP", "FREQ", "FAILED_QA", "BAND", "DOSS") if p.stat().st_size < 3_000_000]
    n = 0
    for src in outs:
        (p,) = RC.copy_into(tmp_path, src)
        ref = S.read_out_frames(p)
        idx = outindex.build(p)                           # live decided from the file: old mtime -> finished
        assert not idx.live
        assert idx.n_steps == len(ref["frames"]), src.name
        assert idx.labels == [f["label"] for f in ref["frames"]], src.name
        assert idx.energies() == [f["energy_au"] for f in ref["frames"]], src.name
        p.unlink()
        n += 1
    assert n > 100


@pytest.mark.skipif(os.environ.get("MACE_GUI_BENCH") != "1", reason="bench: MACE_GUI_BENCH=1")
def test_extend_of_a_big_live_run_is_fast(tmp_path):
    from test_outindex import make_big
    outindex, _, _ = _mods()
    p = tmp_path / "big.out"
    make_big(p)
    idx = outindex.build(p, live=True)
    block = Path(H.EXAMPLES / "outputs" / "Ag2Br3_opt_unfinished.out").read_text().split("\n")[384:457]
    add = ("\n".join(block) + "\n").encode()
    times = []
    for _ in range(3):
        with open(p, "ab") as fh:
            while True:
                fh.write(add)
                if fh.tell() - idx.size >= 50_000:
                    break
        t0 = time.perf_counter()
        idx = outindex.extend(idx, p, live=True)
        times.append(time.perf_counter() - t0)
    p.unlink()
    assert min(times) < 0.1, times


def test_crashed_outputs_end_at_once(tmp_path):
    """An output CRYSTAL stopped with a fatal "ERROR ****" or the MPI launcher ended
    (an out-of-memory kill: "BAD TERMINATION ... KILLED BY SIGNAL: 9") is finished
    as soon as it is written, not "still running" for the 10-minute live window."""
    _, livefile, _ = _mods()
    from mace.gui.submit import out_bucket
    outs = list(RC.corpus_outs("FAILED_QA"))
    assert outs
    for src in outs:
        (p,) = RC.copy_into(tmp_path, src)
        os.utime(p, None)                                   # written just now
        st = livefile.state(p)
        assert st["finished"] and not st["live"], src.name
        assert out_bucket(p) != "ongoing", src.name
        p.unlink()


def test_running_outputs_stay_live(tmp_path):
    """No mid-run line of a corpus output (OPT, SP, FREQ, BAND, DOSS, written up to
    a point) reads as an end: each one cut at a third and at two thirds is live."""
    _, livefile, _ = _mods()
    outs = [p for p in RC.corpus_outs("OPT", "SP", "FREQ", "BAND", "DOSS") if p.stat().st_size < 3_000_000]
    n = 0
    for src in outs:
        data = src.read_bytes()
        for frac in (1 / 3, 2 / 3):
            cut = data[:int(len(data) * frac)]
            if any(m.encode() in cut for m in ("TOTAL CPU TIME =", "EEEEEEEEEE TERMINATION", "OPT END")):
                continue                                    # a finished step inside (a multi-step run)
            p = tmp_path / src.name
            p.write_bytes(cut)
            assert livefile.is_live(p), (src.name, frac, livefile.tail_text(p, livefile.snapshot(p))[-300:])
            n += 1
    assert n > 100
