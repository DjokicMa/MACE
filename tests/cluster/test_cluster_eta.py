"""Nested ETA of a running CRYSTAL job (mace/cluster/progress.py): the
backtest helpers, the held-out gates on the real corpus in test/ (read-only;
truncations are made in memory or in tmp copies) and the levels."""
import collections
import math
from pathlib import Path

import pytest

import _eta_backtest as B


# ------------------------------------------------------------------ T0: helpers
def test_family_split_keeps_duplicates_together():
    """OPT/SP near-duplicates (identical opening SCF energies) and the
    conformers of one composition land in one family, so in one half."""
    B.need_corpus(B.TEST_DATA / "OPT", B.TEST_DATA / "SP")
    fam = B.corpus_families()
    fps = collections.defaultdict(list)
    for p in fam:
        fp = B.scf_fingerprint(Path(p).read_text(errors="replace"))
        if fp:
            fps[fp].append(p)
    dup = [v for v in fps.values() if len(v) > 1]
    assert len(dup) >= 20                         # the corpus' OPT<->SP repeats
    assert all(len({fam[p] for p in v}) == 1 for v in dup)
    confs = collections.defaultdict(set)
    for p, f in fam.items():
        if "-conf" in Path(p).name:
            confs[Path(p).name.split("-conf")[0]].add(f)
    assert confs and all(len(v) == 1 for v in confs.values())
    by_half = collections.Counter(B.half(f) for f in set(fam.values()))
    assert by_half["dev"] >= 15 and by_half["held"] >= 15


def test_name_key_strips_calc_suffixes():
    assert B.name_key("x/1LiFSI-4EC-conf3_MOLECULE_OPT_symm_HSESOL3C.out") == "1lifsi-4ec"
    assert B.name_key("Ag1Cl5_sym_CRYSTAL_OPT_symm_PBE-D3_opt_sp_B3LYP.out") == "ag1cl5"
    assert B.name_key("EC_MOLECULE_OPT_freq_HSESOL3C_optimized_temp.out") == "ec"
    assert B.name_key("3.4^2T137_rev1_sp_B3LYP.out") == B.name_key("3.4^2T137_BULK_OPTGEOM_TZ.out")


def test_bootstrap_ci_brackets_point_estimate():
    vals = [1] * 70 + [0] * 30
    groups = [f"f{i % 20}" for i in range(100)]
    p, lo, hi = B.bootstrap_ci(vals, groups)
    assert p == pytest.approx(0.7)
    assert lo < p < hi and hi - lo < 0.4
    p1, lo1, hi1 = B.bootstrap_ci([1, 0, 1, 1], ["a", "b", "c", "d"])      # one run per family: Wilson
    assert (lo1, hi1) == pytest.approx(B.wilson(3, 4))
    assert B.floor05(0.7349) == 0.70 and B.floor05(0.75) == 0.75
    assert B.subsample(range(100), 30)[0] == 0 and B.subsample(range(100), 30)[-1] == 99
    assert len(B.subsample(range(100), 30)) == 30


# ------------------------------------------------------------------ T1: SCF reader
from mace.cluster import progress as P          # noqa: E402

SP_DIA = B.TEST_DATA / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out"


def _cut(src, dst, n):
    lines = Path(src).read_text(errors="replace").splitlines(True)
    Path(dst).write_text("".join(lines[:n]))
    return Path(dst)


def test_scf_progress_reads_out_convergence_blocks(tmp_path):
    """A real SP cut right after CYC 5 counts 6 cycles through the real
    parser (out_convergence gives SCF blocks, not flat cycles; the shipped
    code saw 0 cycles)."""
    B.need_corpus(SP_DIA)
    lines = SP_DIA.read_text().splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith(" CYC   5 "))
    r = P.progress(_cut(SP_DIA, tmp_path / "sp.out", k + 1))
    assert r["mode"] == "scf" and r["parser"] == "out_convergence"
    assert r["done"] == 6 and r["summary"].startswith("SCF cycle 6")
    flat = P.scf_cycles(P._mini_parse(lines[:k + 1]))
    assert [c["cycle"] for c in flat] == [0, 1, 2, 3, 4, 5]


def _three_sps():
    sps = []
    for f in sorted((B.TEST_DATA / "SP").glob("*.out")):
        t = f.read_text(errors="replace")
        n = t.count("\n CYC ")
        if "OPTIMIZATION - POINT" not in t and 10 <= n <= 60 and len(t) < 3_000_000:
            sps.append(f)
    return sps[:1] + sps[len(sps) // 2:len(sps) // 2 + 1] + sps[-1:]


def test_scf_progress_never_raises(tmp_path):
    """Every truncation of three real SPs, through both parsers: no
    exception (an infinite high end once raised int(inf)), low <= high."""
    B.need_corpus(B.TEST_DATA / "SP")
    sps = _three_sps()
    assert len(sps) == 3
    for src in sps:
        lines = src.read_text(errors="replace").splitlines()
        cuts = [i for i, l in enumerate(lines) if P._CYC.match(l)]
        for j, c in enumerate(cuts):
            sub = lines[:c + 1]
            sp = P.scf_progress(P._mini_parse(sub), sub)
            assert sp["cycles"] == j + 1
            if sp["low_cycles"] is not None and sp["high_cycles"] is not None:
                assert sp["low_cycles"] <= sp["high_cycles"]
            if j % 3 == 0 or j == len(cuts) - 1:
                r = P.progress(_cut(src, tmp_path / "x.out", c + 1))
                assert r["mode"] == "scf" and "error" not in r
                assert r["done"] == j + 1, (src.name, j)


# ------------------------------------------------------------------ T2: OPT points (F2)
def _opt_rows(which):
    """One row per truncation just before POINT k+1 (k = 3..N-1, at most 30
    per run) of the completed multi-point corpus optimizations of one half."""
    rows = []
    for r in B.opt_runs():
        if B.half(r["family"]) != which:
            continue
        lines = r["lines"]
        n = len(P._mini_parse(lines)["points"])
        if n < 4:
            continue
        fr = P.finalrun_from_deck(r["deck"])
        for k in B.subsample(range(3, n), 30):
            sub = lines[:r["heads"][k]]
            parsed = P._mini_parse(sub)
            eta = P.opt_eta(parsed, sub, finalrun=fr)
            lo, hi = eta["low_points"], eta["high_points"]
            ranged = lo is not None
            assert not ranged or hi is None or lo <= hi
            if not ranged:
                assert eta["low_s"] is None and eta["basis"]
            if eta["capped"]:
                assert eta["confidence"] == "rough"          # a prior/MAXCYCLE high end never reads "fair"
            true = n - len(parsed["points"])
            trem = r["t_end"] - B.last_telapse(sub)
            thit = (eta["low_s"] <= trem <= eta["high_s"]) if ranged and eta["low_s"] is not None else None
            rows.append(dict(family=r["family"], run=r["path"], ranged=ranged,
                             hit=ranged and lo <= true <= (hi if hi is not None else 10 ** 9),
                             capped=eta["capped"], low=lo, high=hi, true=true, thit=thit))
    return rows


def test_opt_eta_held_out_gate():
    """The optimization's points left on the family split (whole
    conformer/OPT-SP families in one half).

    The parameters were first picked in a study that scored the whole corpus
    (.planning/polish/ETA_BACKTEST.md), so they were chosen again on the dev
    half alone: grouped 5-fold cross-validation over the 21 dev families, a
    grid of fit window 4/5/6/8, sigma 2/3/4, high floor 3-6x, final-restart
    high +3/+5/+7 and time low factor 0.4-0.75, the prior (high end when
    only MAXCYCLE bounds it) the training folds' 90th percentile of points
    left; rule fixed before the run: unknown <= 35 % and median width <= 10,
    best mean of coverage and time coverage, ties within 0.01 to the
    narrower then to the earlier values. Chosen: window 6, sigma 2, floor 6x,
    +1/+7, time x0.5, prior 44 (the whole dev half's p90). The earlier values
    scored 79 % / time 86 % in the same cross-validation, the chosen ones 83 %
    / 90 %. Held-out, scored once (95 % family-bootstrap intervals):
      dev  420 snapshots, 27 runs, 21 families (in-sample): coverage 85 %
           (78-90), informative 86 % (78-93), time 91 % (86-96), unknown 30 %,
           capped 30 %, width p50 8.2 p90 17.0
      held 341 snapshots, 23 runs, 17 families: coverage 84 % (75-93),
           informative 81 % (68-92), per family 89 % (81-96), time 91 %
           (82-100), unknown 32 %, capped 33 %, width p50 9.0 p90 17.0
    Floors are the held-out lower bounds rounded down to 0.05. The capped
    share (high end = the prior; those ranges always read "rough", asserted
    per row), the unknown share and the widths are regression bounds."""
    B.need_corpus(B.TEST_DATA / "OPT")
    s = B.summarize(_opt_rows("held"))
    print("\nOPT held-out:", B.fmt(s))
    assert s["runs"] >= 20 and s["families"] >= 15
    assert s["coverage"][0] >= 0.75
    assert s["informative"][0] >= 0.65
    assert s["per_family"][0] >= 0.80
    assert s["time"][0] >= 0.80
    assert s["unknown"] <= 0.32 + 0.05
    assert s["capped_share"] <= 0.33 * 1.15
    assert s["width_p50"] <= 8.2 * 1.15 and s["width_p90"] <= 17.0 * 1.15


def test_opt_gate_parser_parity(tmp_path):
    """The gate scores the in-memory reader; progress() on a file copy
    (out_convergence's parse, the deck sibling's FINALRUN) gives the same
    point range on a sample of the same truncations."""
    B.need_corpus(B.TEST_DATA / "OPT")
    checked = 0
    for r in B.opt_runs()[::4]:
        n = len(P._mini_parse(r["lines"])["points"])
        if n < 4:
            continue
        for k in B.subsample(range(3, n), 3):
            sub = r["lines"][:r["heads"][k]]
            d = tmp_path / f"r{checked}"
            d.mkdir()
            (d / "x.out").write_text("\n".join(sub) + "\n")
            if r["deck"]:
                (d / "x.d12").write_text(r["deck"])
            a = P.progress(d / "x.out")["eta"]
            b = P.opt_eta(P._mini_parse(sub), sub, finalrun=P.finalrun_from_deck(r["deck"]))
            assert (a["low_points"], a["high_points"]) == (b["low_points"], b["high_points"]), (r["path"], k)
            checked += 1
    assert checked >= 15


def test_finalrun_from_deck():
    assert P.finalrun_from_deck("") is True                          # CRYSTAL23 default FINALRUN 4
    assert P.finalrun_from_deck("t\nOPTGEOM\nFINALRUN\n4\nENDOPT\n") is True
    assert P.finalrun_from_deck("t\nOPTGEOM\nFINALRUN\n0\nENDOPT\n") is False
    assert P.finalrun_from_deck("t\nOPTGEOM\nfinalrun\n2\nENDOPT\n") is False


# ------------------------------------------------------------------ T3: levels
import os          # noqa: E402
import re          # noqa: E402
import shutil      # noqa: E402
import time        # noqa: E402

OPT_DIA3 = B.TEST_DATA / "OPT" / "3_dia3_opt_rev1.out"
SUPERCELL = B.TEST_DATA / "FREQ" / "1_dia_opt_rev1_freq_B3LYP-D3-D3_optimized_supercel222.out"
SULFOLANE = (B.TEST_DATA / "FREQ" / "1LiFSI-8Sulfolane-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_"
             "HSESOL3C_optimized_freq_HSESOL3C_optimized.out")
LEVEL_KEYS = {"id", "depth", "label", "status", "current", "count", "extra", "per_unit_s", "done_s",
              "remaining_s", "confidence", "unknown_reason", "flags", "thresholds", "basis", "info", "text"}


def _copy(src, d, n=None, name=None, deck=True):
    src = Path(src)
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    lines = src.read_text(errors="replace").splitlines(True)
    out = d / (name or src.name)
    out.write_text("".join(lines if n is None else lines[:n]))
    if deck and src.with_suffix(".d12").exists():
        shutil.copy(src.with_suffix(".d12"), out.with_suffix(".d12"))
    return out


def _check_levels(r):
    lv = r["levels"]
    assert r["levels_version"] == 1 and lv[0]["id"] == "total" and lv[0]["depth"] == 0
    for x in lv:
        assert set(x) == LEVEL_KEYS, x["id"]
        assert x["text"] == P.level_text(x) and x["text"]
        lo, hi = x["remaining_s"]["low"], x["remaining_s"]["high"]
        assert lo is None or hi is None or lo <= hi + 1e-9, x
        assert x["status"] in ("done", "running", "pending", "skipped", "restored")
    assert r["eta"]["low_s"] == lv[0]["remaining_s"]["low"] and r["eta"]["high_s"] == lv[0]["remaining_s"]["high"]
    return lv


def test_levels_shape_all_modes(tmp_path):
    """Real OPT, SP, FREQ and BAND copies (cut mid-run and whole): levels
    with the total first, every key, text only from level_text, low <= high."""
    B.need_corpus(OPT_DIA3, SP_DIA, SUPERCELL)
    band = sorted((B.TEST_DATA / "BAND").glob("*_band.out"))[0]
    cases = [(OPT_DIA3, 1000, "opt"), (OPT_DIA3, None, "opt"), (SP_DIA, 600, "scf"), (SP_DIA, None, "scf"),
             (SUPERCELL, 1040, "freq"), (SUPERCELL, None, "freq"), (band, None, "properties"), (band, 400, None)]
    for k, (src, n, mode) in enumerate(cases):
        r = P.progress(_copy(src, tmp_path / str(k), n))
        assert mode is None or r["mode"] == mode, (src.name, r["mode"])
        lv = _check_levels(r)
        if r["ended"]:
            assert lv[0]["text"].startswith("Total: finished")
        else:
            assert lv[0]["text"].startswith("Total")


def test_eta_keys_backward_compatible(tmp_path):
    """Old keys keep their names and types; the old per-mode sentence moves
    to eta.stage_basis; OPT point keys and freq.basis are unchanged."""
    B.need_corpus(OPT_DIA3, SUPERCELL)
    r = P.progress(_copy(OPT_DIA3, tmp_path / "o", 1000))
    for k in ("mode", "file", "snapshot", "ended", "stage", "done", "total", "percent", "eta", "risk", "summary",
              "criteria", "maxcycle", "parser"):
        assert k in r
    eta = r["eta"]
    for k in ("low_s", "high_s", "basis", "confidence", "low_points", "high_points", "rule", "per_point_s"):
        assert k in eta
    sub = OPT_DIA3.read_text().splitlines()[:1000]
    old = P.opt_eta(P._mini_parse(sub), sub, finalrun=True)
    assert (eta["low_points"], eta["high_points"]) == (old["low_points"], old["high_points"])
    assert eta["stage_basis"] == old["basis"] and "left in total" in eta["basis"]
    lines = SUPERCELL.read_text().splitlines()
    cut = next(i for i, l in enumerate(lines) if "TOTGRA" in l and "TELAPSE" in l) + 1
    f = P.progress(_copy(SUPERCELL, tmp_path / "f", cut))
    assert f["freq"]["basis"].startswith("Central point done (8 min); 5 displacements remaining")
    assert f["eta"]["stage_basis"] == f["freq"]["basis"]


def test_total_at_least_when_a_high_is_unknown():
    a = P._level("x.a", 1, "Stage A", "running", low_s=100.0, high_s=300.0, confidence="rough")
    b = P._level("x.b", 1, "Stage B", "pending", low_s=50.0, high_s=None, unknown_reason="B has no upper end")
    t = P.total_level([a, b])
    assert t["remaining_s"] == {"low": 150.0, "high": None}
    assert P.level_text(t) == "Total at least 2 min left (B has no upper end)"
    t2 = P.total_level([a, P._level("x.c", 1, "C", "pending", low_s=None, high_s=None, unknown_reason="no idea")])
    assert t2["remaining_s"]["low"] == 100.0 and "at least" in P.level_text(t2)
    t3 = P.total_level([P._level("x.d", 1, "D", "running", unknown_reason="nothing yet")])
    assert P.level_text(t3) == "Total: unknown (nothing yet)" and t3["remaining_s"]["low"] is None
    done = P._level("x.e", 1, "E", "done", low_s=0.0, high_s=0.0)
    t4 = P.total_level([done, a])
    assert t4["remaining_s"] == {"low": 100.0, "high": 300.0} and t4["confidence"] == "rough"
    assert P.level_text(t4) == "Total about 1 min - 5 min left (rough)"
    assert P.level_text(P.total_level([a], ended="completeopt")) == "Total: finished (completeopt)"
    inner = P._level("x.i", 2, "Inner", "running", low_s=1e6, high_s=2e6)
    assert P.total_level([a, inner])["remaining_s"] == {"low": 100.0, "high": 300.0}    # inner is inside its stage


def test_fair_needs_narrow_range():
    a = P._level("x.a", 1, "A", "running", low_s=100.0, high_s=250.0, confidence="fair")
    assert P.total_level([a])["confidence"] == "fair"
    b = P._level("x.b", 1, "B", "running", low_s=100.0, high_s=400.0, confidence="fair")
    assert P.total_level([b])["confidence"] == "rough"                     # > FAIR_MAX_WIDTH wide
    eta = {"low_points": 2, "high_points": 9, "low_s": 10.0, "high_s": 90.0, "capped": False,
           "per_point_s": 10.0, "basis": "x"}
    lv = P.opt_levels({"points": [{"segment": 0}] * 5}, eta, [], True, None, None, None, [])[0]
    assert lv["confidence"] == "rough"
    eta.update(high_points=5)
    lv = P.opt_levels({"points": [{"segment": 0}] * 5}, eta, [], True, None, None, None, [])[0]
    assert lv["confidence"] == "fair"
    eta.update(capped=True)
    lv = P.opt_levels({"points": [{"segment": 0}] * 5}, eta, [], True, None, None, None, [])[0]
    assert lv["confidence"] == "rough" and "prior_high" in lv["flags"]


def test_level_texts():
    """The sentences of each level kind (the CLI and the GUI print these)."""
    lv = P._level("opt.points", 1, "Optimization", "running", current=12, low_n=15, high_n=30,
                  extra={"label": "final restart", "low": 1, "high": 7})
    assert P.level_text(lv) == "Optimization point 12 of about 15-30 (+ final restart 1-7)"
    lv["info"]["segment"] = 1
    lv["extra"] = None
    assert P.level_text(lv) == "Final restart point 12 of about 15-30"
    lv = P._level("opt.points", 1, "Optimization", "running", current=2, unknown_reason="2 points done; an estimate needs 3")
    assert P.level_text(lv) == "Optimization point 2 (points left unknown: 2 points done; an estimate needs 3)"
    lv = P._level("opt.points", 1, "Optimization", "running", current=0, unknown_reason="0 points done; an estimate needs 3")
    assert P.level_text(lv) == "Optimization: first point running (points left unknown: 0 points done; an estimate needs 3)"
    lv = P._level("opt.point_scf", 2, "x", "running", current=9, low_n=11, high_n=14, per_unit_s=240)
    assert P.level_text(lv) == "Next point's SCF: cycle 9 of about 11-14, ~4 min per cycle"
    lv = P._level("opt.point_scf", 2, "x", "running", low_n=11, high_n=14,
                  info={"in_point_s": 600, "point_low_s": 1800, "point_high_s": 3600})
    assert P.level_text(lv) == ("Next point's SCF: about 11-14 cycles (from earlier points); running 10 min of "
                                "about 30 min - 1 h 0 min per point")
    lv["info"]["in_point_s"] = 5400                       # past the slowest earlier point: no "of about"
    assert P.level_text(lv) == ("Next point's SCF: about 11-14 cycles (from earlier points); running 1 h 30 min, "
                                "longer than the earlier points (30 min - 1 h 0 min each)")
    lv = P._level("sp.scf", 1, "SCF", "running", current=6, low_n=7, high_n=14, per_unit_s=4.0,
                  info={"detot": 5.5e-9, "toldee": 1e-9})
    assert P.level_text(lv) == "SCF cycle 6 of about 7-14, ~4 s per cycle (|DE| 5.5e-9, threshold 1e-9)"
    lv["flags"] = ["guessp", "maxcycle_cap"]
    lv["info"]["maxcycle"] = 20
    assert P.level_text(lv).endswith("(at most 20 cycles: MAXCYCLE) - restart from a previous density (GUESSP): "
                                     "estimate fit on runs without it, usually fewer cycles")
    lv = P._level("freq.cphf", 1, "x", "running", current=2, exact=3,
                  info={"direction": "Y", "cycle": 4, "delta": -2.5e-2, "tol": 1e-3})
    assert P.level_text(lv) == ("IR/Raman (CPHF/CPKS, first order): field direction Y (2 of 3), cycle 4, "
                                "|DELTA| 2.5e-2 vs 1e-3")
    lv = P._level("freq.displacements", 1, "x", "running", current=3, exact=5, per_unit_s=1200,
                  info={"percent": 60, "restored": 4, "computed": 2})
    assert P.level_text(lv) == ("Displacement 3 of 5 (60 %), about 15 min - 30 min each - 4 restored from "
                                "FREQINFO.DAT, rate from the 2 computed in this run")
    assert P.level_text(P._level("freq.output", 1, "x", "pending", low_s=0.0, high_s=30.0)) == \
        "Frequencies and output: about 30 s or less"
    cen = P._level("freq.central", 1, "x", "running")
    assert P.level_text(cen) == "Central point (SCF + gradient): running, before its first SCF cycle"
    cen["current"] = 0
    assert P.level_text(cen) == "Central point (SCF + gradient): running, SCF cycle 0"
    cen["info"]["restart_setup"] = True
    assert P.level_text(cen) == "Central point: restarting from FREQINFO.DAT"
    assert P._sci(1.5e-4) == "1.5e-4" and P._sci(1e-9) == "1e-9" and P._sci(4.5e-4) == "4.5e-4"


# ------------------------------------------------------------------ T4: OPT levels
def _opt_case_rows(which):
    rows = []
    for r in B.opt_runs():
        if B.half(r["family"]) != which:
            continue
        full = P._mini_parse(r["lines"])["points"]
        if len(full) < 4:
            continue
        for k in B.subsample(range(3, len(full)), 30):
            counts = [p.get("scf_cycles") for p in full[1:k]]
            lo, hi = P._cycle_window(counts)
            true = full[k]["scf_cycles"]
            rows.append(dict(family=r["family"], run=r["path"], ranged=lo is not None,
                             hit=lo is not None and lo <= true <= hi, capped=False, low=lo, high=hi,
                             true=true, thit=None))
    return rows


def test_opt_inner_cycles_held_out_gate():
    """The next point's SCF cycles in [min-1, max+2] of the last 5 points
    (points 2.., chosen on dev among windows 3/5/7 and paddings 0-2; the
    planning had already measured this window on the whole corpus, 84 % over
    1001 transitions, so the held-out figure is not fully blind). After
    point 1 the .out keeps only each point's cycle count (SCF moved to
    SCFOUT.LOG). The level is shown for orientation and adds nothing to the
    total.
      dev  420 truncations, 27 runs, 21 families: 84 % (80-88), width p50 2.5
      held 341 truncations, 23 runs, 17 families: 82 % (78-87), per family 77 % (64-85)"""
    B.need_corpus(B.TEST_DATA / "OPT")
    s = B.summarize(_opt_case_rows("held"))
    print("\nOPT inner held-out:", B.fmt(s))
    assert s["coverage"][0] >= 0.75 and s["per_family"][0] >= 0.60
    assert s["unknown"] <= 0.05 and s["width_p90"] <= 4.3 * 1.15


def test_opt_levels_with_scfout_log(tmp_path):
    """SCFOUT.LOG holding the real CYC lines of point 1 of a corpus OPT, cut
    at cycle 4: the inner level reads "cycle 4 of about"."""
    B.need_corpus(OPT_DIA3)
    lines = OPT_DIA3.read_text().splitlines()
    cyc = [l for l in lines if P._CYC.match(l)]
    scratch = tmp_path / "scr"
    scratch.mkdir()
    (scratch / "SCFOUT.LOG").write_text("\n".join(cyc[:5]) + "\n")
    hs = [i for i, l in enumerate(lines) if P._POINT.search(l)]
    out = _copy(OPT_DIA3, tmp_path / "o", hs[4])
    r = P.progress(out, scratch_dir=scratch)
    inner = [x for x in _check_levels(r) if x["id"] == "opt.point_scf"][0]
    assert inner["current"] == 4 and inner["text"].startswith("Next point's SCF: cycle 4 of about ")


def test_opt_levels_without_scratch(tmp_path):
    B.need_corpus(OPT_DIA3)
    lines = OPT_DIA3.read_text().splitlines()
    hs = [i for i, l in enumerate(lines) if P._POINT.search(l)]
    r = P.progress(_copy(OPT_DIA3, tmp_path / "o", hs[4]), scratch_dir=tmp_path / "missing")
    inner = [x for x in _check_levels(r) if x["id"] == "opt.point_scf"][0]
    assert inner["current"] is None and "cycles (from earlier points)" in inner["text"]
    pts = [x for x in r["levels"] if x["id"] == "opt.points"][0]
    assert pts["text"].startswith("Optimization point 4 of about")


def test_scf_log_tail_tolerates_unknown_format(tmp_path):
    assert P.scf_log_tail(None) is None
    assert P.scf_log_tail(tmp_path) is None                                  # no file
    f = tmp_path / "SCFOUT.LOG"
    for content in ("", "garbage\n\x00\x01 more garbage", " CYC   3 ETOT(AU) -1.0E+00 DETOT -1.0E-05 tst\n",
                    " CYC   0 ETOT(AU) -1.0E+00 DETOT -1.0E+00\n CYC   2 ETOT(AU) -1.0E+00 DETOT -1.0E-02\n"
                    " CYC   1 ETOT(AU) -1.0E+00 DETOT -1.0E-02\n"):
        f.write_text(content)
        assert P.scf_log_tail(tmp_path) is None, content
    f.write_text(" CYC   0 ETOT(AU) -1.0E+00 DETOT -1.0E+00\n CYC   1 ETOT(AU) -1.0E+00 DETOT -1.0E-02\n"
                 " CYC   0 ETOT(AU) -1.0E+00 DETOT -1.0E+00\n CYC   1 ETOT(AU) -1.0E+00 DETOT -1.0E-02\n"
                 " CYC   2 ETOT(AU) -1.0E+00 DETOT -1.0E-03\n CYC   3 ETOT(AU) -1.0")
    assert P.scf_log_tail(tmp_path) == 2                                     # the half-written line is not a cycle


def test_run_start_from_starting_date():
    """CRYSTAL's STARTING line anchors the clock when it is plausible; an
    implausible SLURM start (more than 10 min after it) falls back."""
    B.need_corpus(OPT_DIA3)
    lines = OPT_DIA3.read_text().splitlines()
    m = next(P._STARTING.search(l) for l in lines if P._STARTING.search(l))
    d, mo, y, h, mi, sec = m.groups()
    t0 = time.mktime((int(y), int(mo), int(d), int(h), int(mi), int(float(sec)), 0, 0, -1)) + \
        float(sec) - int(float(sec))
    t, src = P.run_start(lines, mtime=t0 + 3600, slurm_start=t0 - 120)     # module load + staging before CRYSTAL
    assert src == "crystal" and t == pytest.approx(t0)
    t, src = P.run_start(lines, mtime=t0 + 3600, slurm_start=t0 + 3000)    # node clock far behind SLURM's
    assert (t, src) == (t0 + 3000, "slurm")
    t, src = P.run_start([l for l in lines if "STARTING" not in l], mtime=t0, slurm_start=t0 - 5)
    assert (t, src) == (t0 - 5, "slurm")
    assert P.run_start(lines, mtime=t0 - 7200) == (None, None)             # a start after the file's last change


# ------------------------------------------------------------------ T5: thresholds
def _rewrite_thresholds(text, factor):
    def col(m):
        return f"{m.group(1)}{float(m.group(2)) * factor:.6f}"
    text = re.sub(r"(MAXIMUM GRADIENT COMPONENT\s+)(\S+)", col, text)
    text = re.sub(r"(MAX GRADIENT\s+\S+\s+THRESHOLD\s+)(\S+)", col, text)
    text = re.sub(r"(THRESHOLD ON ENERGY CHANGE\s+)(\S+)",
                  lambda m: f"{m.group(1)}{float(m.group(2)) * factor:.3E}", text)
    return text


def test_threshold_note_tighter_and_looser(tmp_path):
    """A real OPT copy with the max-gradient thresholds and the energy-change
    threshold rewritten x0.1 / x10: the notes name the CRYSTAL23 default."""
    B.need_corpus(OPT_DIA3)
    lines = OPT_DIA3.read_text().splitlines(True)
    hs = [i for i, l in enumerate(lines) if P._POINT.search(l)]
    base = "".join(lines[:hs[4]])
    for factor, word in ((0.1, "tighter"), (10, "looser")):
        d = tmp_path / word
        d.mkdir()
        (d / "x.out").write_text(_rewrite_thresholds(base, factor))
        r = P.progress(d / "x.out")
        notes = {n["name"]: n for n in [x for x in r["levels"] if x["id"] == "opt.points"][0]["thresholds"]}
        assert notes["max gradient"]["note"] == f"{word} than the default 4.5e-4"
        assert notes["energy change"]["note"] == f"{word} than the default 1e-7"
        assert notes["RMS gradient"]["note"] is None
        line = P.thresholds_line(list(notes.values()))
        assert line.startswith(f"Thresholds: max gradient {P._sci(4.5e-4 * factor)} ({word} than the default 4.5e-4)")


def test_threshold_note_uses_context_default():
    sp = P.threshold_notes({"toldee_scf": {"value": 1e-9}}, "sp")
    assert sp[0]["note"] == "tighter than the single-point default 1e-6"
    assert P.threshold_notes({"toldee_scf": {"value": 1e-9}}, "freq")[0]["note"] is None
    assert P.threshold_notes({"toldee_scf": {"value": 1e-7}}, "freq")[0]["note"] == "looser than the FREQCALC default 1e-9"
    opt = P.threshold_notes({"toldee_scf": {"value": 1e-7}}, "opt")
    assert opt[0]["note"] is None
    assert P.thresholds_line(opt) is None and P.thresholds_line(opt, always=True) == "Thresholds: CRYSTAL23 defaults"


def test_threshold_display_does_not_change_eta(tmp_path):
    """Display only: the estimate of a prior-capped case (the first one in
    the corpus) is identical with and without a tighter threshold in the
    header."""
    B.need_corpus(B.TEST_DATA / "OPT")
    sub = next(r["lines"][:r["heads"][k]] for r in B.opt_runs() for k in range(3, len(r["heads"]))
               if P.opt_eta(P._mini_parse(r["lines"][:r["heads"][k]]), r["lines"][:r["heads"][k]])["capped"])
    base = "\n".join(sub) + "\n"
    tight = re.sub(r"(MAXIMUM GRADIENT COMPONENT\s+)(\S+)", lambda m: m.group(1) + "0.00015", base)
    assert tight != base
    (tmp_path / "a.out").write_text(base)
    (tmp_path / "b.out").write_text(tight)
    a, b = P.progress(tmp_path / "a.out"), P.progress(tmp_path / "b.out")
    for k in ("low_points", "high_points", "low_s", "high_s", "capped"):
        assert a["eta"][k] == b["eta"][k], k
    assert a["eta"]["capped"]


# ------------------------------------------------------------------ T6: SP levels
def _sp_rows(which, time_only=False):
    rows = []
    for r in B.sp_runs():
        if B.half(r["family"]) != which or r["guessp"]:
            continue
        for k in B.subsample(range(3, r["n"]), 30):
            sub = r["lines"][:r["ci"][k]]
            sp = P.scf_progress(P._mini_parse(sub), sub)
            assert sp["cycles"] == k
            lo, hi = sp["low_cycles"], sp["high_cycles"]
            assert lo is None or lo <= hi
            if sp["capped"]:
                assert sp["confidence"] == "rough"
            true = r["n"] - k
            ranged = lo is not None
            trem = r["t_end"] - B.last_telapse(sub)
            thit = (sp["low_s"] <= trem <= sp["high_s"]) if ranged and sp["low_s"] is not None else None
            rows.append(dict(family=r["family"], run=r["path"], ranged=ranged,
                             hit=ranged and lo <= true <= hi, capped=sp["capped"], low=lo, high=hi, true=true,
                             thit=thit, sp=sp, trem_total=r["t_last"] - B.last_telapse(sub)))
    return rows


def test_scf_eta_held_out_gate():
    """SP SCF cycles left: running minimum of log10(|DE|/TOLDEE), 10-cycle
    fit at 3 standard errors, high end >= 8x the low end, an open high end
    -> 30 cycles (dev p90). Chosen by grouped 5-fold CV over dev families
    among the shipped rule (windows 3/6/10, sigma 3/5), its running-minimum
    variant, the conditional prior and the combinations (best coverage with
    unknown <= 35 % and capped <= 25 %); held-out scored once.
    As chosen (the selection script):
      dev  375 snapshots, 23 runs, 20 families: coverage 88 % (84-91), time 89 %
      held 678 snapshots, 34 runs, 19 families: coverage 72 % (61-91),
           informative 75 % (66-91), per family 87 % (83-90), time 73 % (62-94),
           unknown 40 %, capped 14 %
    The selection script scored a cut whose |DE| was already below TOLDEE as
    0 cycles left, but CRYSTAL needs two such cycles in a row, so exactly one
    is left (23 of 23 dev cuts); the implementation gives 1-2 cycles there
    (0 s low end). This was found on dev after the first held-out scoring:
    dev 95 % (91-99), time 95 %; held 80 % (70-100), informative 85 %
    (76-100), per family 97 % (93-100), time 80 % (69-100), unknown 40 %,
    capped 13 %, width p50 8.0 p90 15.0.
    The held-out half holds the long slab SCFs (200+ cycles, erratic |DE|):
    most of its misses (all too short) and its unknowns. Floors = the lower
    bounds of the first held-out scoring rounded down to 0.05; unknown and
    widths are regression bounds. The 2 GUESSP outputs are left out (flagged
    live instead).
    1.1.6: a run that has printed "POSSIBLY CONDUCTING STATE" gets its high
    end x METAL_HIGH_FACTOR (1.5, the smallest of 1/1.5/2/3 with 100 %
    informative coverage on the dev metals; held-out metals 74 % -> 89 %,
    insulators unchanged at 95 %). Held-out overall: coverage 91 % (86-100),
    informative 94 %, time 90 %, width p50 12.0 p90 22.5 (the width gates
    follow the wider metal ranges)."""
    B.need_corpus(B.TEST_DATA / "SP")
    s = B.summarize(_sp_rows("held"))
    print("\nSCF held-out:", B.fmt(s))
    assert s["coverage"][0] >= 0.60 and s["informative"][0] >= 0.65 and s["per_family"][0] >= 0.80
    assert s["time"][0] >= 0.60
    assert s["unknown"] <= 0.40 + 0.05 and s["capped_share"] <= 0.25
    assert s["width_p50"] <= 12.0 * 1.15 and s["width_p90"] <= 22.5 * 1.15


def test_metal_scf_high_end_is_widened():
    """The held-out metals (runs that printed POSSIBLY CONDUCTING STATE): 22 runs, 9 families,
    coverage 89 % (85-100) with the widening, 74 % without; the floor is the lower bound rounded down."""
    B.need_corpus(B.TEST_DATA / "SP")
    metal = [r for r in _sp_rows("held") if any("POSSIBLY CONDUCTING STATE" in l for l in
                                                 next(x for x in B.sp_runs() if x["path"] == r["run"])["lines"])]
    assert metal and all("metal" in r["sp"]["flags"] or not r["ranged"] or r["sp"].get("capped") is not None for r in metal)
    s = B.summarize(metal)
    assert s["coverage"][0] >= 0.85, B.fmt(s)


def test_sp_total_close_to_scf():
    """The SP total (SCF + the post-SCF margin, to the file's last timing
    line) covers about as often as the SCF time alone: the CIs overlap."""
    B.need_corpus(B.TEST_DATA / "SP")
    rows = _sp_rows("held")
    tot = []
    for r in rows:
        sp = r["sp"]
        if sp["low_s"] is None:
            continue
        post = max(10.0, 0.01 * sp["high_s"])
        tot.append(dict(r, thit=sp["low_s"] <= r["trem_total"] <= sp["high_s"] + post))
    a, b = B.summarize(rows)["time"], B.summarize(tot)["time"]
    print(f"\nSP time {a[0]:.0%} ({a[1]:.0%}-{a[2]:.0%}), total {b[0]:.0%} ({b[1]:.0%}-{b[2]:.0%})")
    assert b[1] <= a[2] and a[1] <= b[2]


def test_scf_gate_parser_parity(tmp_path):
    """progress() on file copies (out_convergence) counts the same cycles
    and gives the same range as the in-memory reader the gate scores."""
    B.need_corpus(B.TEST_DATA / "SP")
    n = 0
    for r in B.sp_runs()[::6]:
        for k in B.subsample(range(3, r["n"]), 2):
            sub = r["lines"][:r["ci"][k]]
            d = tmp_path / f"s{n}"
            d.mkdir()
            (d / "x.out").write_text("\n".join(sub) + "\n")
            a = P.progress(d / "x.out")["scf"]
            b = P.scf_progress(P._mini_parse(sub), sub)
            assert (a["cycles"], a["low_cycles"], a["high_cycles"]) == (b["cycles"], b["low_cycles"],
                                                                         b["high_cycles"]) or \
                "maxcycle_cap" in a["flags"], r["path"]
            n += 1
    assert n >= 8


def test_guessp_deck_is_flagged(tmp_path):
    """GUESSP in the deck sibling, or CRYSTAL's GUESSP line in the .out:
    the SCF level is flagged and says the estimate is fit on runs without it."""
    B.need_corpus(SP_DIA)
    lines = SP_DIA.read_text().splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith(" CYC   5 "))
    out = _copy(SP_DIA, tmp_path / "a", k + 1, deck=False)
    plain = [x for x in P.progress(out)["levels"] if x["id"] == "sp.scf"][0]
    assert "guessp" not in plain["flags"]
    deck = SP_DIA.with_suffix(".d12")
    d12 = deck.read_text() if deck.exists() else "title\nCRYSTAL\nEND\n"
    out.with_suffix(".d12").write_text(d12.replace("END", "GUESSP\nEND", 1) if "GUESSP" not in d12 else d12)
    scf = [x for x in P.progress(out)["levels"] if x["id"] == "sp.scf"][0]
    assert "guessp" in scf["flags"] and "(GUESSP): estimate fit on runs without it" in scf["text"]
    real = [f for f in sorted((B.TEST_DATA / "SP").glob("*.out"))
            if "SCF GUESS FROM A PREVIOUS RUN DENSITY MATRIX" in f.read_text(errors="replace")]
    if real:
        rl = real[0].read_text(errors="replace").splitlines()
        cut = [i for i, l in enumerate(rl) if P._CYC.match(l)][4]
        r = P.progress(_copy(real[0], tmp_path / "b", cut + 1, deck=False))
        assert "guessp" in [x for x in r["levels"] if x["id"] == "sp.scf"][0]["flags"]


def test_scf_high_capped_by_maxcycle():
    """The deck's SCF MAXCYCLE caps the high end (CRYSTAL stops there)."""
    B.need_corpus(SP_DIA)
    lines = SP_DIA.read_text().splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith(" CYC   5 "))
    sub = lines[:k + 1]
    parsed = P._mini_parse(sub)
    free = P.scf_progress(parsed, sub)
    assert free["high_cycles"] > 3
    parsed["thresholds"]["scf_maxcycle"] = {"value": 8}
    capped = P.scf_progress(parsed, sub)
    assert capped["high_cycles"] == 8 - 6 and "maxcycle_cap" in capped["flags"]
    lv = P.sp_levels(capped, sub, "\n".join(sub), "", None, [])[1]
    assert "(at most 8 cycles: MAXCYCLE)" in P.level_text(lv)


def test_scf_one_more_cycle_once_below_toldee():
    """CRYSTAL ends the SCF after two cycles in a row with |DE| < TOLDEE: a
    real SP cut right after the first such cycle has exactly one cycle left
    and gets 1-2 (time from 0, the last cycle's work may be all but done),
    not the trend's 8x floor; a bounce above TOLDEE after a dip gets 1-8."""
    B.need_corpus(SP_DIA)
    lines = SP_DIA.read_text().splitlines()
    tol = P.scf_progress(P._mini_parse(lines), lines)["toldee"]
    ci = [i for i, l in enumerate(lines) if P._CYC.match(l)]
    k = next(j for j, i in enumerate(ci) if abs(P._f(P._CYC.match(lines[i]).group(3))) < tol)
    assert k == len(ci) - 2                                   # the real run: one more cycle, then SCF ENDED
    sub = lines[:ci[k] + 1]
    sp = P.scf_progress(P._mini_parse(sub), sub)
    assert (sp["low_cycles"], sp["high_cycles"]) == (1, 2) and sp["confidence"] == "fair"
    assert sp["low_s"] == 0.0 and sp["high_s"] > 0
    bounce = sub + [P._CYC.sub(lambda m: m.group(0), sub[-1]).replace(P._CYC.match(sub[-1]).group(3), "-5.00E-03")]
    sp = P.scf_progress(P._mini_parse(bounce), bounce)
    assert (sp["low_cycles"], sp["high_cycles"]) == (1, 8)


def test_sp_levels_stage_order(tmp_path):
    """Setup -> SCF -> after the SCF: running, done and pending in turn."""
    B.need_corpus(SP_DIA)
    lines = SP_DIA.read_text().splitlines()
    c0 = next(i for i, l in enumerate(lines) if P._CYC.match(l))
    end = next(i for i, l in enumerate(lines) if "== SCF ENDED" in l)
    for n, want in ((c0 - 5, ("running", "pending", "pending")), (c0 + 150, ("done", "running", "pending")),
                    (end + 2, ("done", "done", "running")), (None, ("done", "done", "done"))):
        r = P.progress(_copy(SP_DIA, tmp_path / str(n), n))
        lv = _check_levels(r)
        assert tuple(x["status"] for x in lv[1:]) == want, (n, [x["text"] for x in lv])


# ------------------------------------------------------------------ T7: FREQ levels
def _ids(r):
    return [x["id"] for x in r["levels"][1:]]


def _lv(r, lid):
    return next(x for x in r["levels"] if x["id"] == lid)


def test_freq_cphf_order_on_corpus(tmp_path):
    """Every corpus FREQ that runs CPHF (CP_DRIVER): reference SCF -> CPKS
    (-> CP2, Raman) -> tensors -> central point -> displacements -> output;
    the central point's time leaves the CPHF stages out."""
    B.need_corpus(B.TEST_DATA / "FREQ")
    seen = 0
    for f in sorted((B.TEST_DATA / "FREQ").glob("*.out")):
        text = f.read_text(errors="replace")
        if "CP_DRIVER" not in text:
            continue
        r = P.progress(_copy(f, tmp_path / str(seen), deck=False))
        ids = _ids(r)
        want = ["freq.ref_scf", "freq.cphf"] + (["freq.cp2"] if "CP2_DRIVER" in text else []) + \
            ["freq.tensors", "freq.central", "freq.displacements", "freq.output"]
        assert ids == want, (f.name, ids)
        assert all(x["status"] == "done" for x in r["levels"][1:])
        seen += 1
    assert seen == 9
    r = P.progress(_copy(SULFOLANE, tmp_path / "s", deck=False))
    cen = _lv(r, "freq.central")["done_s"]
    first_totgra = next(t for (_, n, t) in P.timings(SULFOLANE.read_text().splitlines()) if n == "TOTGRA")
    assert cen < 0.5 * first_totgra and 1500 < cen < 2500          # about 2006 s of 18771 s


def test_freq_8sulfolane_cut_in_cpks(tmp_path):
    B.need_corpus(SULFOLANE)
    r = P.progress(_copy(SULFOLANE, tmp_path, 2850, deck=False))
    lv = _check_levels(r)
    assert r["stage"] == "cphf"
    assert _lv(r, "freq.ref_scf")["status"] == "done" and _lv(r, "freq.ref_scf")["done_s"] < 2000
    cp = _lv(r, "freq.cphf")
    assert cp["status"] == "running" and cp["info"]["direction"] == "Y" and cp["current"] == 2
    assert "field direction Y (2 of 3), cycle " in cp["text"] and "vs 1e-3" in cp["text"]
    assert _lv(r, "freq.central")["status"] == "pending"
    dsp = _lv(r, "freq.displacements")
    assert dsp["status"] == "pending" and dsp["text"] == "Displacements: pending"    # count not printed yet
    assert dsp["basis"] == ""                                  # the CPKS sentence is not the displacements' reason
    # 1.1.6: the CPKS directions left, the tensors and the central point have ranges; the total waits only
    # for the displacements' time per displacement
    assert cp["count"]["low"] is not None and cp["remaining_s"]["high"] > cp["remaining_s"]["low"] > 0
    assert "cycles (" in cp["text"] and " left, ~" in cp["text"] and cp["confidence"] == "rough"
    cen = _lv(r, "freq.central")
    assert cen["remaining_s"]["low"] is not None and cen["text"].startswith("Central point (SCF + gradient): pending, about")
    assert lv[0]["text"].startswith("Total at least ") and "time per displacement unknown" in lv[0]["text"]
    assert r["eta"]["low_s"] is not None and r["eta"]["high_s"] is None
    # once the frequency header is printed the displacement count is exact
    lines = SULFOLANE.read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if "NUMBER OF SCF+GRADIENT" in l)
    r = P.progress(_copy(SULFOLANE, tmp_path / "h", hdr + 1, deck=False))
    assert _lv(r, "freq.displacements")["count"]["exact"] == 387
    assert _lv(r, "freq.central")["status"] == "running" or r["stage"] == "central"


def test_freq_cphf_run_after_the_table_is_frequencies(tmp_path):
    """After the last displacement of a CPHF run (its IR/Raman stages ran
    before the central point) the stage is the frequencies, and the
    heading and the total agree that little is left."""
    B.need_corpus(SULFOLANE)
    lines = SULFOLANE.read_text().splitlines()
    last_row = max(i for i, l in enumerate(lines) if P._DISP_ROW.match(l))
    nxt = next(i for i in range(last_row, len(lines)) if P._TELAPSE.match(lines[i]))
    r = P.progress(_copy(SULFOLANE, tmp_path, nxt + 1, deck=False))
    _check_levels(r)
    assert r["stage"] == "frequencies" and "intensities" not in r["summary"].lower(), r["summary"]
    assert _lv(r, "freq.output")["status"] == "running"
    assert r["levels"][0]["text"].startswith("Total about"), r["levels"][0]["text"]


def test_freq_rows_without_a_rate_say_why(tmp_path):
    """Rows finished but no time per displacement: the line names the
    reason instead of "until the first finishes"."""
    B.need_corpus(SULFOLANE)
    lines = SULFOLANE.read_text().splitlines()
    rows = [i for i, l in enumerate(lines) if P._DISP_ROW.match(l)]
    r = P.progress(_copy(SULFOLANE, tmp_path, rows[4] + 1, deck=False))          # no job: no clock
    d = _lv(r, "freq.displacements")
    assert d["current"] == 5 and d["per_unit_s"] is None
    assert d["text"] == "Displacement 5 of 387 (1 %): time per displacement unknown (no job clock to time it " \
                        "(a copied or finished output))", d["text"]
    assert r["levels"][0]["text"].startswith("Total: unknown (time per displacement unknown: no job clock")


def test_freq_8sulfolane_cut_in_central(tmp_path):
    B.need_corpus(SULFOLANE)
    r = P.progress(_copy(SULFOLANE, tmp_path, 5000, deck=False))
    _check_levels(r)
    assert r["stage"] == "central"
    cen = _lv(r, "freq.central")
    assert cen["status"] == "running" and cen["text"] == "Central point (SCF + gradient): running, gradient"
    for lid in ("freq.cphf", "freq.cp2", "freq.tensors", "freq.ref_scf"):
        x = _lv(r, lid)
        assert x["status"] == "done" and x["done_s"] and x["done_s"] > 0, lid
    assert _lv(r, "freq.cp2")["done_s"] > _lv(r, "freq.cphf")["done_s"]
    assert r["freq"]["basis"] == "Central point (SCF + gradient) running"


def _partial_restart_lines():
    """A computed FREQ (9 rows, real lines) turned into a restart killed and
    resubmitted mid-table: its central SCF block and TOTGRA removed, the
    restart lines of a real FREQINFO.DAT output inserted where CRYSTAL prints
    them, the central row's N.CYC 0 and the first 4 rows in the real
    RESTORED format."""
    comp = B.TEST_DATA / "FREQ" / "3,4^2T10-CA_rev1_freq_B3LYP-D3-D3_optimized.out"
    B.need_corpus(comp)                 # before the search below: skip, not StopIteration, without test/
    rest = next(f for f in sorted((B.TEST_DATA / "FREQ").glob("*.out"))
                if "RESTORED FROM A PREV. HESS. MATRIX" in f.read_text(errors="replace"))
    L = comp.read_text(errors="replace").splitlines()
    R = rest.read_text(errors="replace").splitlines()
    info = next(l for l in R if "FREQUENCY CALCULATION RESTART(DATA FROM FREQINFO.DAT)" in l)
    first = next(l for l in R if "THE FIRST POINT IS RESTORED FROM THE FREQINFO.DAT" in l)
    incompl = next(l for l in R if "RESTART FROM  AN INCOMPLETE HESSIAN MATRIX" in l)
    restored_fmt = next(l for l in R if "RESTORED FROM A PREV. HESS. MATRIX" in l)
    hdr = next(i for i, l in enumerate(L) if "NUMBER OF SCF+GRADIENT" in l)
    c0 = next(i for i, l in enumerate(L) if P._CYC.match(l))
    totgra = next(i for i, l in enumerate(L) if "TOTGRA" in l and P._TELAPSE.match(l))
    table = next(i for i, l in enumerate(L) if "MAX ABS(DGRAD)" in l and "ATOM" in l and "TOTAL ENERGY" in l)
    out = L[:hdr] + [info] + L[hdr:c0] + ["", first, "", incompl, ""] + L[totgra + 1:table + 1]
    rows = []
    for l in L[table + 1:]:
        if "CENTRAL POINT" in l:
            parts = l.split()
            rows.append(l.replace(f" {parts[3]} ", " 0 ".rjust(len(parts[3]) + 2), 1))
        elif P._DISP_ROW.match(l):
            if sum(1 for x in rows if "RESTORED" in x) < 4:
                m = re.match(r"^(\s+\d+\s+\S+\s+D[XYZ])", l)
                rows.append(m.group(1) + restored_fmt[len(m.group(1)):] if len(restored_fmt) > len(m.group(1))
                            else l)
            else:
                rows.append(l)
        else:
            break
    return out, rows


def test_freq_partial_restart(tmp_path):
    """Restored central point + 4 restored rows + computed rows: cut after
    computed row 6 -> central "restored", done 6, the rate from the 2 rows
    computed in this run, anchored at this run's last timing line before
    the table (the shipped code had no anchor: no TOTGRA in a restart)."""
    head, rows = _partial_restart_lines()
    assert sum("RESTORED" in r for r in rows) == 4 and rows[0].split()[3] == "0"
    lines = head + rows[:1 + 6]
    f = tmp_path / "r.out"
    f.write_text("\n".join(lines) + "\n")
    r = P.progress(f)
    _check_levels(r)
    assert r["mode"] == "freq" and r["stage"] == "displacements" and r["done"] == 6
    assert _lv(r, "freq.central")["status"] == "restored"
    assert "freq.cphf" not in _ids(r)
    fp = r["freq"]
    assert fp["restart"] and (fp["restored"], fp["computed"]) == (4, 2)
    tl = P.timings(lines)
    anchor = tl[-1][2]                                   # the last timing line, before the table
    mtime = os.stat(f).st_mtime
    start = mtime - (anchor + 2 * 1800)                  # as if the 2 computed rows took 30 min each
    fp = P.freq_progress(f.read_text(), lines, False, mtime=mtime, job_start=start)
    assert fp["per_displacement_s"] == pytest.approx(1800, rel=0.01)
    assert fp["low_s"] == pytest.approx(3 * 1800 * 0.75, rel=0.01)
    assert "(4 restored from FREQINFO.DAT)" in fp["basis"]
    lv = P.freq_levels(fp, P.freq_stages(lines), lines, None, None, [])
    dsp = next(x for x in lv if x["id"] == "freq.displacements")
    assert "4 restored from FREQINFO.DAT, rate from the 2 computed in this run" in P.level_text(dsp)


def test_freq_full_restart_counts_unchanged(tmp_path):
    """The corpus' all-restored FREQINFO.DAT outputs: finished, every row
    counted, the central point restored, no CPHF stage invented."""
    B.need_corpus(B.TEST_DATA / "FREQ")
    n = 0
    for f in sorted((B.TEST_DATA / "FREQ").glob("*.out")):
        text = f.read_text(errors="replace")
        if "RESTORED FROM A PREV. HESS. MATRIX" not in text or n >= 25:
            continue
        r = P.progress(_copy(f, tmp_path / str(n)))
        k = text.count("RESTORED FROM A PREV. HESS. MATRIX")
        assert r["stage"] == "end" and r["freq"]["restored"] == k and r["done"] == min(k, r["total"])
        assert _lv(r, "freq.central")["status"] == "restored"
        assert not {"freq.cphf", "freq.ref_scf"} & set(_ids(r))
        n += 1
    assert n >= 20


def test_freq_mid_displacements_keeps_range_and_risk(tmp_path):
    """A computed run cut mid-table with a running job: a total range and a
    risk level (the total never turns unknown because of the last stage)."""
    B.need_corpus(SUPERCELL)
    lines = SUPERCELL.read_text().splitlines()
    head = next(i for i, l in enumerate(lines) if "MAX ABS(DGRAD)" in l and "TOTAL ENERGY" in l)
    rows = [i for i in range(head + 2, len(lines)) if P._DISP_ROW.match(lines[i])]
    f = _copy(SUPERCELL, tmp_path, rows[1] + 1)
    r = P.progress(f, job={"elapsed_s": 555.03 + 2 * 3400 + 60, "limit_s": 4 * 3600})
    lv = _check_levels(r)
    assert r["eta"]["low_s"] is not None and r["eta"]["high_s"] is not None
    assert lv[0]["text"].startswith("Total about ")
    assert r["risk"]["level"] in ("ok", "tight", "likely to exceed")
    assert _lv(r, "freq.output")["remaining_s"]["high"] >= 30


def _freq_cost_rows(which):
    row = re.compile(r"^\s+\d+\s+\S+\s+D[XYZ]\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)")
    out = []
    fam = B.corpus_families()
    for f in sorted((B.TEST_DATA / "FREQ").glob("*.out")):
        if B.half(fam[str(f)]) != which:
            continue
        cyc = [int(m.group(3)) for m in (row.match(l) for l in f.read_text(errors="replace").splitlines()) if m]
        n = len(cyc)
        if n < 2:
            continue
        for k in B.subsample(range(1, n), 30):
            per = sum(cyc[:k]) / k
            lo, hi = per * (n - k) * P.FREQ_LOW_FACTOR, per * (n - k) * P.FREQ_HIGH_FACTOR
            true = sum(cyc[k:])
            out.append(dict(family=fam[str(f)], run=str(f), ranged=True, hit=lo <= true <= hi, capped=False,
                            low=lo, high=hi, true=true, thit=None))
    return out


def test_freq_displacement_cost_held_out_gate():
    """Rows carry no time, so the displacement rule (rows left x the mean so
    far x 0.75-1.5) is backtested in SCF cycles (N.CYC) after row k.
      dev  97 snapshots, 17 runs, 14 families: 88 % (74-96)
      held 111 snapshots, 11 runs, 11 families: 89 % (63-100), per family 76 % (53-100)
    The shipped factors stay (dev is above the held-out floor). Weighting
    cycles by the central/row SYM ratio lost coverage on dev (68 %), so
    it is not used."""
    B.need_corpus(B.TEST_DATA / "FREQ")
    s = B.summarize(_freq_cost_rows("held"))
    print("\nFREQ held-out:", B.fmt(s))
    assert s["runs"] >= 8 and s["coverage"][0] >= 0.60 and s["per_family"][0] >= 0.50


def test_freq_preopt_stage_name(tmp_path):
    """A FREQ with a preoptimization (synthesized from real lines: a FREQ's
    opening, then real optimization points) names the stage and the point."""
    B.need_corpus(SUPERCELL, OPT_DIA3)
    F = SUPERCELL.read_text().splitlines()
    O = OPT_DIA3.read_text().splitlines()
    hdr = next(i for i, l in enumerate(F) if "NUMBER OF SCF+GRADIENT" in l)
    hs = [i for i, l in enumerate(O) if P._POINT.search(l)]
    lines = F[:hdr - 5] + O[hs[0]:hs[3]]
    f = tmp_path / "p.out"
    f.write_text("\n".join(lines) + "\n")
    r = P.progress(f)
    assert r["mode"] == "freq" and r["stage"] == "preopt"
    pre = _lv(r, "freq.preopt")
    assert pre["status"] == "running" and pre["text"] == "Preoptimization: running (optimization point 3)"


# ------------------------------------------------------------------ T7b: refresh cost
def test_progress_size_budget(tmp_path):
    """A 50 MB .out made by repeating a real SP's SCF cycle blocks: one
    refresh costs at most 1.3 x the shipped passes (read + calc-type
    detection + out_convergence parse + end-marker check) + 0.5 s; an
    unchanged snapshot comes from the memo in < 0.05 s."""
    B.need_corpus(SP_DIA)
    lines = SP_DIA.read_text().splitlines(True)
    c0 = next(i for i, l in enumerate(lines) if P._CYC.match(l))
    end = next(i for i, l in enumerate(lines) if "== SCF ENDED" in l)
    head, block = "".join(lines[:c0]), "".join(lines[c0 + 1:end - 1])
    f = tmp_path / "big.out"
    with open(f, "w") as fh:
        fh.write(head)
        size = len(head)
        while size < 50 * 1024 * 1024:
            fh.write(block)
            size += len(block)
    from mace.utils import out_convergence
    t = time.perf_counter()
    text, snap = P.read_text(f)
    P.mode_of(text)
    out_convergence.parse(f, size=snap["size"], complete_lines_only=True)
    P.ended_state(f)
    shipped = time.perf_counter() - t
    P._MEMO.clear()
    t = time.perf_counter()
    r = P.progress(f)
    new = time.perf_counter() - t
    t = time.perf_counter()
    r2 = P.progress(f)
    hit = time.perf_counter() - t
    print(f"\n50 MB: shipped passes {shipped:.2f} s, progress() {new:.2f} s, memo hit {hit * 1000:.1f} ms")
    assert r["mode"] == "scf" and r2 == r
    assert new <= 1.3 * shipped + 0.5
    assert hit < 0.05


def test_memo_follows_the_file(tmp_path):
    B.need_corpus(SP_DIA)
    f = _copy(SP_DIA, tmp_path, 600)
    a = P.progress(f)
    a["risk"]["sentence"] = "changed by a caller"
    assert P.progress(f)["risk"]["sentence"] is None              # callers get copies
    with open(f, "a") as fh:
        fh.write(SP_DIA.read_text().splitlines(True)[600])
    assert P.progress(f)["snapshot"]["size"] > a["snapshot"]["size"]


# ------------------------------------------------------------------ T9: walltime risk
def _risk_cases(which):
    out = []
    for r in B.opt_runs():
        if B.half(r["family"]) != which:
            continue
        n = len(P._mini_parse(r["lines"])["points"])
        if n < 4:
            continue
        for k in B.subsample(range(3, n), 30):
            sub = r["lines"][:r["heads"][k]]
            eta = P.opt_eta(P._mini_parse(sub), sub, finalrun=P.finalrun_from_deck(r["deck"]))
            out.append((r["family"], eta["low_s"], eta["high_s"], r["t_end"] - B.last_telapse(sub)))
    for r in B.sp_runs():
        if B.half(r["family"]) != which or r["guessp"]:
            continue
        for k in B.subsample(range(3, r["n"]), 30):
            sub = r["lines"][:r["ci"][k]]
            sp = P.scf_progress(P._mini_parse(sub), sub)
            hi = None if sp["high_s"] is None else sp["high_s"] + max(10.0, 0.01 * sp["high_s"])
            out.append((r["family"], sp["low_s"], hi, r["t_last"] - B.last_telapse(sub)))
    return out


def test_risk_labels_backtest_gate():
    """Walltime labels on OPT and SP truncations with a synthetic walltime
    left W = R x {0.5, 0.8, 1.25, 2, 4} (R = the true time left). Dev, with
    the "tight" factor 0.8 (0.6/0.8/1.0 tried): false "likely to exceed"
    0.08 (0.02-0.14), "tight" on a clearly safe job 0.13 (0.10-0.18), missed
    overrun 0.02 (0.01-0.03), all within the targets 0.10/0.20/0.15.
    Held-out (with the optimization parameters chosen again on dev, which left
    the dev numbers unchanged): 0.04 (0.00-0.07), 0.10 (0.08-0.14), 0.06
    (0.00-0.10); gated at the upper bounds rounded up to 0.05."""
    B.need_corpus(B.TEST_DATA / "OPT")
    likely, safe, missed = [], [], []
    for fam, lo, hi, R in _risk_cases("held"):
        if lo is None or R <= 0:
            continue
        for m in (0.5, 0.8, 1.25, 2, 4):
            W = R * m
            lvl, note = P.risk_from_total(0.0, W, {"remaining_s": {"low": lo, "high": hi}})
            if lvl == "likely to exceed":
                likely.append((R <= W, fam))
            if R <= 0.5 * W:
                safe.append((lvl == "tight", fam))
            if R > W:
                missed.append((lvl == "ok" and not note, fam))
    ci = [B.bootstrap_ci([x for x, _ in v], [g for _, g in v]) for v in (likely, safe, missed)]
    print("\nrisk held-out: false likely %.2f (%.2f-%.2f), tight when safe %.2f (%.2f-%.2f), "
          "missed overrun %.2f (%.2f-%.2f)" % (*ci[0], *ci[1], *ci[2]))
    assert len(likely) >= 30 and len(safe) >= 300
    assert ci[0][0] <= 0.10 and ci[1][0] <= 0.15 and ci[2][0] <= 0.10


def test_wide_range_gives_note_not_tight():
    """A wide range whose long end does not fit: a note, not "tight"; the
    central estimate decides "tight"; the low end decides "likely"."""
    tot = {"remaining_s": {"low": 600.0, "high": 36000.0}}
    lvl, note = P.risk_from_total(0.0, 7200.0, tot)            # c = 4650 s < 0.8 x 7200
    assert lvl == "ok" and note.startswith("Could exceed the walltime if it runs to the long end of the range "
                                           "(10 h 0 min left in total, 2 h 0 min of walltime remaining)")
    assert P.risk_from_total(0.0, 5000.0, tot)[0] == "tight"
    assert P.risk_from_total(0.0, 500.0, tot)[0] == "likely to exceed"
    lvl, note = P.risk_from_total(0.0, 1000.0, {"remaining_s": {"low": 600.0, "high": None}})
    assert lvl == "ok" and "at least 10 min left in total" in note       # "at least" totals still warn
    assert P.risk_from_total(0.0, 1e5, tot) == ("ok", None)
    assert P.risk_from_total(None, 1e5, tot) == (None, None)


def test_first_point_scf_level(tmp_path):
    """Inside an optimization's first point the SCF runs in the output itself
    (SCFOUT.LOG only from the second point): its cycle and |DE| against the
    deck's threshold, with no cycles-left estimate; gone once point 1 is
    printed."""
    B.need_corpus(OPT_DIA3)
    lines = OPT_DIA3.read_text(errors="replace").splitlines(True)
    cyc = [i for i, l in enumerate(lines) if P._CYC.match(l)]
    p = _copy(OPT_DIA3, tmp_path, cyc[2] + 1)
    r = P.progress(p)
    ids = [lv["id"] for lv in r["levels"]]
    assert ids == ["total", "opt.points", "opt.first_scf"]
    lv = r["levels"][2]
    assert lv["depth"] == 2 and lv["current"] == 2
    assert lv["count"]["low"] is None and lv["remaining_s"] == {"low": None, "high": None}
    assert re.fullmatch(r"First point's SCF: cycle 2 \(\|DE\| \S+, threshold 1e-\d+\)", lv["text"]), lv["text"]
    assert r["levels"][0]["text"].startswith("Total: unknown (0 points done")
    head = next(i for i, l in enumerate(lines) if P._POINT.search(l))
    r = P.progress(_copy(OPT_DIA3, tmp_path, head + 40, name="b.out"))
    assert "opt.first_scf" not in [lv["id"] for lv in r["levels"]]


def test_total_basis_does_not_repeat_the_reason():
    stage = P._level("opt.points", 1, "Optimization", "running", unknown_reason="2 points done; an estimate needs 3",
                     basis="2 points done; an estimate needs 3")
    other = P._level("freq.output", 1, "Frequencies and output", "pending", low_s=0.0, high_s=30.0,
                     basis="at most 30 s")
    tot = P.total_level([stage])
    assert tot["basis"] == "time left unknown (2 points done; an estimate needs 3)"
    tot = P.total_level([stage, other])
    assert tot["basis"].endswith(": Frequencies and output: at most 30 s")
    assert tot["basis"].count("an estimate needs 3") == 1


FREQ_RAMAN = B.TEST_DATA / "FREQ" / ("1LiFSI-8Sulfolane-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_"
                                     "optimized_freq_HSESOL3C_optimized.out")


def test_running_cphf_stage_says_how_long_it_has_run(tmp_path):
    """The IR tensors print three lines and then nothing for many minutes (a
    live MgO IR/Raman job: over 10 min). Without an estimate the stage says
    how long it has run, from the job's clock; a copied output (no clock)
    says nothing of it."""
    B.need_corpus(FREQ_RAMAN)
    lines = FREQ_RAMAN.read_text(errors="replace").splitlines()
    st = P.freq_stages(lines)
    k = st["tensors"][-1]
    t = P._t_at(P.timings(lines), k)
    p = _copy(FREQ_RAMAN, tmp_path, k + 1, deck=False)
    r = P.progress(p, job={"elapsed_s": t + 600, "limit_s": 86400})
    lv = next(x for x in r["levels"] if x["id"] == "freq.tensors")
    assert lv["status"] == "running"
    assert lv["info"]["so_far_s"] == pytest.approx(600, abs=5)
    # 1.1.6: and its upper bound (TENSOR_HIGH_RATIO x the CPHF stages' time)
    assert lv["text"].startswith("IR/Raman tensors: running (10 min so far), at most about "), lv["text"]
    r = P.progress(_copy(FREQ_RAMAN, tmp_path, k + 1, name="copy.out", deck=False))
    lv = next(x for x in r["levels"] if x["id"] == "freq.tensors")
    assert lv["text"].startswith("IR/Raman tensors: running, at most about ") and "so_far_s" not in lv["info"]
    # a first-order CPHF direction in progress: its time in the stage, after the cycle
    j = st["cp_dirs"][1][0] + 3
    t = P._t_at(P.timings(lines), st["cp_driver"])
    r = P.progress(_copy(FREQ_RAMAN, tmp_path, j, name="cp.out", deck=False), job={"elapsed_s": t + 125})
    lv = next(x for x in r["levels"] if x["id"] == "freq.cphf")
    assert lv["status"] == "running" and " (2 min in this stage)" in lv["text"], lv["text"]


def test_scf_log_tail_none_between_scf_end_and_next_cycle_0(tmp_path):
    """Live FREQ run: after a displacement's SCF ends, CRYSTAL computes its
    gradient for minutes before the next "CYC 0"; the finished SCF's last
    cycle (12) was shown as the next displacement's running cycle. The lines
    are the shape of that run's SCFOUT.LOG."""
    f = tmp_path / "SCFOUT.LOG"
    run = "".join(f" CYC  {c:2d} ETOT(AU) -2.000185519232E+03 DETOT -1.11E-11 tst  5.17E-15 PX  3.85E-08\n"
                  for c in range(13))
    f.write_text(run + " == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -2.0001855192319E+03 CYCLES  12\n"
                 " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT TOTGRA      TELAPSE      833.66 TCPU      831.99\n")
    assert P.scf_log_tail(tmp_path) is None
    f.write_text(f.read_text() + " CYC   0 ETOT(AU) -2.000185646949E+03 DETOT -2.00E+03 tst  5.17E-15 PX  1.00E+00\n")
    assert P.scf_log_tail(tmp_path) == 0
    f.write_text(run)
    assert P.scf_log_tail(tmp_path) == 12


def test_untracked_risk_sentence_fits_the_calc_type():
    """A FREQ job near its walltime was told to restart "an optimization ...
    (OPTGEOM RESTART with OPTINFO.DAT)"; a frequency job continues with
    FREQCALC RESTART from FREQINFO.DAT, a single point only with more time."""
    s = P.risk_sentence("tight", False, {}, 0, False, "FREQ", "/w")
    assert "FREQCALC" in s and "FREQINFO.DAT" in s and "OPTGEOM" not in s
    s = P.risk_sentence("tight", False, {}, 0, False, "OPT", "/w")
    assert "OPTGEOM RESTART with OPTINFO.DAT" in s
    s = P.risk_sentence("likely to exceed", False, {}, 0, False, "SCF", "/w")
    assert "OPTGEOM" not in s and "FREQ" not in s and "longer walltime" in s
    assert P.risk_sentence("ok", False, {}, 0, False, "FREQ", "/w") is None


def test_freq_time_in_the_running_displacement_comes_off_the_low_end(tmp_path):
    """Live rutile FREQ: 3 of 4 rows done, the estimate stayed "6-13 min
    left" for 9 minutes while the last displacement ran, and the job ended a
    minute later. The output does not change during a displacement, so the
    time since its last change is spent in the running one: the low end
    drops by it (never below the displacements after it), the high end
    stays, as for optimization points."""
    head, rows = _partial_restart_lines()
    lines = head + rows[:1 + 6]
    f = tmp_path / "r.out"
    f.write_text("\n".join(lines) + "\n")
    anchor = P.timings(lines)[-1][2]
    mtime = os.stat(f).st_mtime
    start = mtime - (anchor + 2 * 1800)                  # 2 computed rows of 30 min
    plain = P.freq_progress(f.read_text(), lines, False, mtime=mtime, job_start=start)
    left = plain["total"] - plain["done"]
    assert plain["low_s"] == pytest.approx(left * 1800 * 0.75, rel=0.01) and "in_displacement_s" not in plain
    fp = P.freq_progress(f.read_text(), lines, False, mtime=mtime, job_start=start, now=mtime + 1000)
    assert fp["in_displacement_s"] == pytest.approx(1000)
    assert fp["low_s"] == pytest.approx(left * 1800 * 0.75 - 1000, rel=0.01)
    assert fp["high_s"] == plain["high_s"]
    fp = P.freq_progress(f.read_text(), lines, False, mtime=mtime, job_start=start, now=mtime + 10 ** 6)
    assert fp["low_s"] == pytest.approx((left - 1) * 1800 * 0.75, rel=0.01)
    lv = P.freq_levels(fp, P.freq_stages(lines), lines, None, None, [])
    dsp = next(x for x in lv if x["id"] == "freq.displacements")
    assert "; the next running for " in P.level_text(dsp)


def test_sp_scf_unknown_text_says_it_once(tmp_path):
    """Live single point, cycle 1: "SCF cycle 1 (cycles left unknown: SCF
    cycles left unknown: fewer than 3 cycles)". Said once now."""
    B.need_corpus(SP_DIA)
    lines = SP_DIA.read_text(errors="replace").splitlines(True)
    cyc = [i for i, l in enumerate(lines) if P._CYC.match(l)]
    r = P.progress(_copy(SP_DIA, tmp_path, cyc[0] + 1))
    scf = next(x for x in r["levels"] if x["id"] == "sp.scf")
    assert scf["text"] == "SCF cycle 1 (SCF cycles left unknown: fewer than 3 cycles)", scf["text"]


# ------------------------------------------------------------------ 1.1.6: CPHF/CPKS, tensors, central point
def test_cp_cycle_line_without_a_space_after_alpha():
    """A large polarizability runs into the label ("ALPHA26707.745616", corpus Ag1Cl5): the cycle still counts."""
    m = P._CP_CYCLE.match(" CYCLE   3   ALPHA30113.804129  EPSILON  382.389370  DELTA    5.9910E+02")
    assert m and m.group(1) == "3" and float(m.group(2)) == 599.1
    assert P._CP_CYCLE.match(" CYCLE   2  PSEUDO-ENERGY DERIVATIVE        5.913090E+04  DELTA    3.5969E+04")


def test_cphf_estimate_backtest(monkeypatch):
    """Leave-one-family-out over the 9 corpus FREQ runs with CPHF (9 families: 8 Ag halides with
    IR, 1 LiFSI-sulfolane cluster with IR + Raman): the prior cycles per direction and the
    central/reference ratio come from the other families only. Measured (snapshots, Wilson 95 %):
    CPKS cycles left 0.79 (0.73-0.84, 187 cuts), CPKS time left 0.84 (0.78-0.89), tensors bound
    0.90 (0.85-0.93), central point 0.86 (0.80-0.90); CP2 has one run (21 cuts, all covered).
    Gates: the Wilson lower bound rounded down to 0.05. Thin data: the levels stay "rough"."""
    import tempfile
    runs = [p for p in sorted((B.TEST_DATA / "FREQ").glob("*.out")) if "CP_DRIVER" in p.read_text(errors="replace")]
    if len(runs) < 9:
        pytest.skip("test/ corpus not present (gitignored)")
    fam = B.family_of(runs)
    info = {}
    for p in runs:
        L = p.read_text(errors="replace").splitlines()
        st, tl = P.freq_stages(L), P.timings(L)
        cnt = P.cp_estimate(st, L, tl, "cphf", running=False)["counts"] + P.cp_estimate(st, L, tl, "cp2", running=False)["counts"]
        ratio = (P._t_at(tl, st["totgra"]) - P._t_at(tl, st["central_c0"])) / (P._t_at(tl, st["cp_driver"]) - P._t_at(tl, st["ref_c0"]))
        info[p] = (L, st, tl, cnt, ratio)
    hits = collections.defaultdict(list)
    with tempfile.TemporaryDirectory() as td:
        for p, (L, st, tl, _, _) in info.items():
            oth = [v for q, v in info.items() if fam[str(q)] != fam[str(p)]]
            oc = [c for v in oth for c in v[3]]
            monkeypatch.setattr(P, "CP_PRIOR_CYCLES", (min(oc), max(oc)))
            rs = [v[4] for v in oth]
            monkeypatch.setattr(P, "CENTRAL_REF_LOW", round(min(rs) * 0.9, 2))
            monkeypatch.setattr(P, "CENTRAL_REF_HIGH", round(max(rs) * 1.15, 2))
            cp_end = st["cp2_driver"] if st["cp2_driver"] is not None else (st["tensors"][0] if st["tensors"] else st["header"])
            ten0 = st["tensors"][0] if st["tensors"] else st["header"]
            cyc = [c[0] for c in st["cp_cycles"] if c[0] < cp_end]
            for k in B.subsample(list(range(len(cyc))), 30):
                cut = cyc[k] + 1
                q = Path(td) / p.name
                q.write_text("\n".join(L[:cut]) + "\n")
                P._MEMO.clear()
                lv = {x["id"]: x for x in P.progress(q)["levels"]}
                now = P._t_at(tl, cut - 1)
                x = lv["freq.cphf"]
                truth_c = sum(1 for c in st["cp_cycles"] if cut <= c[0] < cp_end)
                hits["cycles"].append(x["count"]["low"] <= truth_c <= x["count"]["high"])
                if x["remaining_s"]["low"] is not None:
                    hits["time"].append(x["remaining_s"]["low"] - 1e-6 <= P._t_at(tl, cp_end) - now <= x["remaining_s"]["high"] + 1e-6)
                y = lv["freq.tensors"]
                if y["remaining_s"]["high"] is not None:
                    hits["tensors"].append(P._t_at(tl, st["header"]) - P._t_at(tl, ten0) <= y["remaining_s"]["high"] + 1e-6)
                z = lv["freq.central"]
                hits["central"].append(z["remaining_s"]["low"] <= P._t_at(tl, st["totgra"]) - P._t_at(tl, st["central_c0"])
                                       <= z["remaining_s"]["high"])
                assert x["confidence"] == ("rough" if x["remaining_s"]["low"] is not None else "unknown")
                assert z["confidence"] == "rough"
    floors = {"cycles": 0.70, "time": 0.75, "tensors": 0.80, "central": 0.80}
    for k, v in hits.items():
        lo, _ = B.wilson(sum(v), len(v))
        assert lo >= floors[k], (k, sum(v), len(v), lo)


# ------------------------------------------------------------------ 1.1.6: properties blocks
_PROP_PAIRS = {"DOSS": ("NEWK", "DOSS", "DOSS"), "ECH3POT3": ("ECH3", "POT3", "POT3"),
               "TRANSPORT": ("NEWK", "BOLTZTRA", "BOLTZTRA")}


def _prop_ratios():
    out = {}
    for d, (anchor, last, key) in _PROP_PAIRS.items():
        for p in sorted((B.TEST_DATA / d).glob("*.out")):
            tl = P.timings(p.read_text(errors="replace").splitlines())
            t = {}
            for _, n, tt in tl:
                t.setdefault(n, tt)
            if not all(k in t for k in ("gordsh1", anchor, last)) or t[anchor] - t["gordsh1"] <= 0.05:
                continue
            out.setdefault(key, []).append((p, (t[last] - t[anchor]) / (t[anchor] - t["gordsh1"])))
    return out


def test_properties_ratio_held_out():
    """A properties block's time from the block before it (PROP_RATIOS = the ratio's min-max on the dev
    half, family split). Held-out coverage: DOSS 36/36 runs (95 % Wilson 0.90-1.00), POT3 26/28
    (0.77-0.98); BOLTZTRA has 4 runs (2 + 2), reported only. Gates: the Wilson lower bounds rounded
    down to 0.05; the constants must bracket the dev range they were read from."""
    data = _prop_ratios()
    if len(data.get("DOSS", [])) < 20:
        pytest.skip("test/ corpus not present (gitignored)")
    fam = B.family_of([p for v in data.values() for p, _ in v])
    floors = {"DOSS": 0.90, "POT3": 0.75}
    for key, v in data.items():
        lo_c, hi_c = P.PROP_RATIOS[key][1:]
        dev = [r for p, r in v if B.half(fam[str(p)]) == "dev"]
        assert lo_c <= min(dev) and hi_c >= max(dev) and hi_c <= max(dev) * 1.1, (key, min(dev), max(dev))
        if key in floors:
            ho = [lo_c <= r <= hi_c for p, r in v if B.half(fam[str(p)]) != "dev"]
            assert B.wilson(sum(ho), len(ho))[0] >= floors[key], (key, sum(ho), len(ho))


def test_properties_levels_on_corpus_cuts(tmp_path):
    import shutil
    doss = sorted((B.TEST_DATA / "DOSS").glob("*.out"))
    ech = sorted((B.TEST_DATA / "ECH3POT3").glob("*.out"))
    if not doss or not ech:
        pytest.skip("test/ corpus not present (gitignored)")
    for src, mark, ids in ((doss[0], "NEWK        TELAPSE", ["properties.newk", "properties.doss"]),
                           (ech[0], "ECH3        TELAPSE", ["properties.ech3", "properties.pot3"])):
        lines = src.read_text(errors="replace").splitlines()
        k = next(i for i, l in enumerate(lines) if mark in l)
        d = tmp_path / src.parent.name
        d.mkdir()
        (d / "x.out").write_text("\n".join(lines[:k + 1]) + "\n")
        shutil.copyfile(src.with_suffix(".d3"), d / "x.d3")
        r = P.progress(d / "x.out")
        assert r["mode"] == "properties"
        lv = [x for x in r["levels"] if x["depth"] == 1]
        assert [x["id"] for x in lv] == ids
        assert lv[0]["status"] == "done" and lv[0]["text"].endswith(lv[0]["text"].split(": done in ")[-1])
        assert lv[1]["status"] == "running" and lv[1]["confidence"] == "rough"
        assert lv[1]["remaining_s"]["high"] > lv[1]["remaining_s"]["low"] >= 0
        # a range from 0 s reads "under N" (1.1.6; it read "about 0 s - N")
        assert r["levels"][0]["text"].startswith(("Total about ", "Total under ")), r["levels"][0]["text"]
        # before the first block finishes: the total is unknown and says why
        (d / "y.out").write_text("\n".join(lines[:next(i for i, l in enumerate(lines) if "gordsh1" in l) + 1]) + "\n")
        shutil.copyfile(src.with_suffix(".d3"), d / "y.d3")
        r = P.progress(d / "y.out")
        assert r["mode"] == "properties" and r["levels"][0]["text"].startswith("Total: unknown (")
        assert "no earlier block to scale its time from" in r["levels"][0]["text"]


# ------------------------------------------------------------------ 1.1.6: EOS, ELASTCON, ANHARM
def _repeated_opt(n: int, src=None):
    """n optimizations one after the other (a real corpus OPT, its TELAPSE shifted each time): the shape
    an EOS (one V-constrained optimization per volume) or ELASTCON run prints. No such run exists in the
    corpus, so the unit markers are CRYSTAL's ordinary OPT END lines."""
    src = src or (B.TEST_DATA / "OPT" / "1_dia_opt_rev1.out")
    B.need_corpus(src)
    lines = src.read_text(errors="replace").splitlines()
    start = next(i for i, l in enumerate(lines) if "OPTIMIZATION - POINT" in l) - 40
    end = next(i for i, l in enumerate(lines) if "OPT END - CONVERGED" in l) + 1
    head, body = lines[:start], lines[start:end]
    tl = P.timings(lines)
    span = P._t_at(tl, end) - P._t_at(tl, start)
    out = list(head)
    pat = re.compile(r"(TELAPSE\s+)([\d.]+)")
    for k in range(n):
        out += [pat.sub(lambda m: f"{m.group(1)}{float(m.group(2)) + k * span:.2f}", l) for l in body]
    return out, span


def test_eos_volumes_from_the_deck_range(tmp_path):
    lines, span = _repeated_opt(2)
    (tmp_path / "e.d12").write_text("x\nCRYSTAL\n0 0 0\n227\n3.5\n1\n6 0 0 0\nEOS\nRANGE\n0.92 1.08 5\nEND\nEND\n")
    (tmp_path / "e.out").write_text("\n".join(lines) + "\n")
    r = P.progress(tmp_path / "e.out")
    assert r["mode"] == "eos" and r["total"] == 5 and r["done"] == 2
    lv = [x for x in r["levels"] if x["id"] == "eos.units"][0]
    assert lv["current"] == 3 and lv["count"]["exact"] == 5 and lv["confidence"] == "rough" and "untested" in lv["flags"]
    # 3 volumes left x the measured time per volume x 0.75-1.5, less the time already in the third
    assert lv["remaining_s"]["high"] == pytest.approx(3 * span * 1.5, rel=0.05)
    assert lv["text"].startswith("Equation of state (EOS): volume 3 of 5, about ")
    assert "no run of this kind in the corpus" in lv["basis"]
    assert r["levels"][0]["text"].startswith("Total about ")
    # the default 6 volumes without RANGE; before the first volume finishes the time is unknown
    (tmp_path / "f.d12").write_text("x\nCRYSTAL\nEOS\nEND\nEND\n")
    (tmp_path / "f.out").write_text("\n".join(lines[:len(lines) // 3]) + "\n")
    r = P.progress(tmp_path / "f.out")
    lv = [x for x in r["levels"] if x["id"] == "eos.units"][0]
    assert lv["count"]["exact"] == 6 and lv["remaining_s"]["low"] is None
    assert "time per volume unknown until the first finishes" in lv["text"]


def test_elastcon_counts_need_deform(tmp_path):
    lines, _ = _repeated_opt(2)
    (tmp_path / "a.out").write_text("\n".join(lines) + "\n")
    (tmp_path / "a.d12").write_text("x\nCRYSTAL\nELASTCON\nEND\nEND\n")
    lv = [x for x in P.progress(tmp_path / "a.out")["levels"] if x["id"] == "elastcon.units"][0]
    assert lv["count"]["exact"] is None and "symmetry analysis" in lv["text"]
    assert P.progress(tmp_path / "a.out")["levels"][0]["text"].startswith("Total: unknown (configurations left unknown")
    (tmp_path / "b.out").write_text("\n".join(lines) + "\n")
    (tmp_path / "b.d12").write_text("x\nCRYSTAL\nELASTCON\nDEFORM\n1 0 0 1 0 0\nNUMDERIV\n5\nEND\nEND\n")
    lv = [x for x in P.progress(tmp_path / "b.out")["levels"] if x["id"] == "elastcon.units"][0]
    assert lv["count"]["exact"] == 2 * 4 + 1 and lv["remaining_s"]["high"] is not None


def test_anharm_points(tmp_path):
    B.need_corpus(B.TEST_DATA / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out")
    sp = (B.TEST_DATA / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out").read_text().splitlines()
    (tmp_path / "h.out").write_text("\n".join(sp) + "\n")
    (tmp_path / "h.d12").write_text("x\nCRYSTAL\nANHARM\n2\nPOINTS26\nEND\nEND\n")
    r = P.progress(tmp_path / "h.out")
    assert r["mode"] == "anharm" and r["total"] == 26
    lv = [x for x in r["levels"] if x["id"] == "anharm.units"][0]
    assert lv["current"] == 2 and lv["remaining_s"]["low"] is not None


# ------------------------------------------------------------------ 1.1.6: possibly stalled
def test_stall_threshold_against_the_corpus():
    """No finished corpus run was ever silent for STALL_FACTOR x its time per unit: the longest silence
    between two timing lines was 5.35 x the median of the 5 SCF cycles (SP) or points (OPT) before it on
    dev (94 runs) and 6.85 x on held-out (74 runs), so 10 x gives no false alarm on either half."""
    import statistics as stats
    runs = sorted((B.TEST_DATA / "SP").glob("*.out")) + sorted((B.TEST_DATA / "OPT").glob("*.out"))
    if len(runs) < 50:
        pytest.skip("test/ corpus not present (gitignored)")
    cyc = re.compile(r"^\s*CYC\s+\d+")
    worst = 0.0
    for f in runs:
        L = f.read_text(errors="replace").splitlines()
        tl = P.timings(L)
        units = [i for i, l in enumerate(L) if (cyc.match(l) if f.parent.name == "SP" else "OPTIMIZATION - POINT" in l)]
        ut = [P._t_at(tl, i) for i in units]
        for k in range(6, len(units)):
            prev = [b - a for a, b in zip(ut[k - 6:k - 1], ut[k - 5:k]) if a is not None and b is not None]
            typ = stats.median(prev) if prev else 0
            if typ <= 0.5:
                continue
            ts = [t for i, _, t in tl if units[k - 1] <= i <= units[k]]
            if len(ts) > 1:
                worst = max(worst, max(b - a for a, b in zip(ts, ts[1:])) / typ)
    assert 0 < worst < P.STALL_FACTOR, worst


def test_possibly_stalled_flag(tmp_path):
    src = B.TEST_DATA / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out"
    B.need_corpus(src)
    lines = src.read_text().splitlines()
    k = [i for i, l in enumerate(lines) if l.lstrip().startswith("CYC ")][6]
    p = tmp_path / "s.out"
    p.write_text("\n".join(lines[:k + 1]) + "\n")
    old = time.time() - 4 * 3600                      # written 4 h ago, cycles take seconds
    os.utime(p, (old, old))
    r = P.progress(p, job={"elapsed_s": 5 * 3600, "limit_s": 86400})
    assert r.get("stalled") and "possibly_stalled" in r["levels"][0]["flags"]
    assert " - possibly stalled: no output for 4 h" in r["levels"][0]["text"], r["levels"][0]["text"]
    # a fresh write, or a copied output without the job's clock: no flag
    os.utime(p, None)
    P._MEMO.clear()
    assert not P.progress(p, job={"elapsed_s": 5 * 3600, "limit_s": 86400}).get("stalled")
    os.utime(p, (old, old))
    P._MEMO.clear()
    assert not P.progress(p).get("stalled")
