"""Edited band k-paths (Crystal_d3/d3_band_path.py) and the BAND decks they give.

* the segment operations the CLI prompt and the GUI editor share;
* decks with no edit are byte-identical to the decks written before band_path
  existed (tests/fixtures/band_path_golden, captured from dev/116-gui d12fef39);
* an edited path is written as the edit says, in coordinates or labels.
"""
import copy
import io
import contextlib
import shutil
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
import d3_band_path as kp  # noqa: E402

GOLDEN = REPO_ROOT / "tests" / "fixtures" / "band_path_golden"
BASE = {"calculation_type": "BAND", "n_points": 1000, "bands": "auto", "auto_bands": True,
        "first_band": 1, "last_band": None, "plot": True, "print_eigenvalues": False}
CONFIGS = {
    "labels": {"path_method": "labels", "path": "auto", "auto_path": True, "shrink": 0},
    "coords": {"path_method": "coordinates", "path": "auto", "auto_path": True, "shrink": "auto"},
    "seekpath": {"path_method": "coordinates", "path": "auto", "auto_path": True, "seekpath_full": True,
                 "shrink": "auto"},
    "literature": {"path_method": "coordinates", "path": "auto", "auto_path": True, "literature_path": True,
                   "shrink": "auto"},
}
INPUTS = {"si": "examples/gui/outputs/Si_sp.out", "graphene_slab": "examples/gui/outputs/graphene_slab_sp.out",
          "dia": "test/OPT/1_dia_opt_rev1.out"}

# graphite's path as the plot reads it: Γ-M-K-Γ-A-L-H-A|L-M|H-K
GRAPHITE = ["GAMMA", "M", "K", "GAMMA", "A", "L", "H", "A", "|", "L", "M", "|", "H", "K"]


def _deck(monkeypatch, tmp_path, src, cfg):
    import CRYSTALOptToD3 as d3gen
    monkeypatch.setattr(d3gen.D3Generator, "_copy_wavefunction", lambda self: True)
    work = tmp_path / "w"
    work.mkdir(exist_ok=True)
    shutil.copy(src, work / src.name)
    out = tmp_path / "o"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        g = d3gen.D3Generator(str(work / src.name), "BAND", output_dir=str(out))
        g.generate_d3(copy.deepcopy(cfg))
    decks = sorted(out.glob("*.d3")) if out.exists() else []
    return (decks[0].read_text() if decks else None), buf.getvalue(), g


def _source(rel):
    p = REPO_ROOT / rel
    if not p.exists():
        pytest.skip(f"{rel} is not here (the test/ corpus is gitignored)")
    return p


# ---------------------------------------------------------------- golden: no edit, no change
@pytest.mark.parametrize("inp", list(INPUTS))
@pytest.mark.parametrize("cname", list(CONFIGS))
def test_unedited_decks_are_byte_identical(monkeypatch, tmp_path, inp, cname):
    golden = GOLDEN / f"{inp}_{cname}.d3"
    src = _source(INPUTS[inp])
    cfg = {**BASE, **CONFIGS[cname]}
    deck, log, gen = _deck(monkeypatch, tmp_path, src, cfg)
    assert deck == golden.read_text(), log[-2000:]
    # the computed path spelled out as band_path is no edit either
    computed = kp.parse_band_deck(golden.read_text())["segments"]
    assert computed
    shutil.rmtree(tmp_path / "o")
    deck2, log, _ = _deck(monkeypatch, tmp_path, src, {**cfg, "band_path": computed})
    assert deck2 == golden.read_text(), log[-2000:]


# ---------------------------------------------------------------- operations
def test_tokens_round_trip_and_pieces():
    segs = kp.from_tokens(GRAPHITE)
    assert len(segs) == 9 and kp.path_tokens(segs) == GRAPHITE
    assert kp.pieces(segs) == [[0, 1, 2, 3, 4, 5, 6], [7], [8]]
    assert kp.canonical_label("Γ") == "GAMMA" and kp.canonical_label("F<sub>0</sub>") == "F_0"
    assert kp.canonical_label("$Δ_{0}$") == "DELTA_0" and kp.canonical_label("G") == "GAMMA"


def test_the_five_step_task():
    """Hide K→Γ, reverse Γ→M, move A|L first, remove H, reset."""
    segs = kp.from_tokens(GRAPHITE)
    s = kp.toggle_hidden(segs, 2)                                    # K→Γ hidden
    assert s[2]["hidden"] and kp.path_tokens(s) == ["GAMMA", "M", "K", "|", "GAMMA", "A", "L", "H", "A",
                                                    "|", "L", "M", "|", "H", "K"]
    s = kp.reverse(s, 0)                                             # M→Γ
    assert (s[0]["from"], s[0]["to"]) == ("M", "GAMMA")
    s = kp.move_piece(s, kp.pieces(s).index([7]), 0)                 # the piece after A|L first
    assert (s[0]["from"], s[0]["to"]) == ("L", "M")
    h = [i for i, x in enumerate(s) if x["to"] == "H"][0]
    s = kp.remove_point(s, h, "to", join=False)                     # H on a plot: L→H and H→A hidden
    assert len(s) == 9 and [(x["from"], x["to"]) for x in s if x.get("hidden")][-2:] == [("L", "H"), ("H", "A")]
    t = kp.path_tokens(s)
    assert any(t[k:k + 3] == ["L", "|", "H"] for k in range(len(t)))         # a jump where H was
    assert kp.from_tokens(GRAPHITE) == segs                          # reset: the computed path


def test_remove_point_joins_for_a_calculation_and_ends_drop_one_segment():
    segs = kp.from_tokens(["GAMMA", "X", "M", "GAMMA"], [[0, 0, 0, .5, 0, 0], [.5, 0, 0, .5, .5, 0], [.5, .5, 0, 0, 0, 0]])
    j = kp.remove_point(segs, 0, "to", join=True)
    assert kp.path_tokens(j) == ["GAMMA", "M", "GAMMA"] and j[0]["k_to"] == [.5, .5, 0]
    e = kp.remove_point(segs, 0, "from", join=True)
    assert kp.path_tokens(e) == ["X", "M", "GAMMA"]


def test_jumps_and_continuous_only():
    segs = kp.from_tokens(GRAPHITE)
    assert kp.path_tokens(kp.jumps_first(segs))[:5] == ["L", "M", "|", "H", "K"]
    assert kp.path_tokens(kp.jumps_last(segs)) == GRAPHITE
    assert kp.path_tokens(kp.continuous_only(segs)) == GRAPHITE[:8]


def test_add_point_and_validation():
    segs = kp.from_tokens(["GAMMA", "X"], [[0, 0, 0, .5, 0, 0]])
    s = kp.add_point(segs, "P", [0.25, 0.25, 0])
    assert s[-1] == {"from": "X", "to": "P", "k_from": [.5, 0, 0], "k_to": [.25, .25, 0]}
    assert kp.validate(s, segs) == ([], [])
    assert kp.validate([dict(segs[0], hidden=True)])[0] == ["At least one segment must stay visible"]
    err, _ = kp.validate([{"from": "GAMMA", "to": "Q?"}])
    assert err and "not a usable label" in err[0]
    err, _ = kp.validate([{"from": "X", "to": "X"}])
    assert err and "no length" in err[0]
    err, _ = kp.validate([{"from": "GAMMA", "to": "Z"}], segs, need_k=True)
    assert err and "give its coordinates" in err[0]
    _, warn = kp.validate([segs[0]], kp.from_tokens(["GAMMA", "X", "M"]))
    assert warn and "no longer visits M" in warn[0]
    assert kp.validate([{"from": "A", "to": "B"}] * 21)[0][0].startswith("BAND takes at most 20")


# ---------------------------------------------------------------- edited decks
def test_edited_coordinate_deck(monkeypatch, tmp_path):
    src = _source(INPUTS["si"])
    cfg = {**BASE, **CONFIGS["seekpath"]}
    golden = (GOLDEN / "si_seekpath.d3").read_text()
    computed = kp.parse_band_deck(golden)["segments"]
    main = kp.continuous_only(computed)
    edit = kp.add_point(kp.reverse(main, 0), "P", [0.25, 0.25, 0.25])
    deck, log, _ = _deck(monkeypatch, tmp_path, src, {**cfg, "band_path": [
        {k: v for k, v in s.items() if not k.startswith("k_") or s["to"] == "P"} for s in edit]})
    assert deck is not None, log[-1500:]
    p = kp.parse_band_deck(deck)
    assert " (edited) - " in p["title"] and p["title"].endswith("-".join(kp.path_tokens(edit)))
    assert [(s["from"], s["to"]) for s in p["segments"]] == [(s["from"], s["to"]) for s in edit]
    assert p["segments"][0]["k_from"] == pytest.approx(main[0]["k_to"])     # reversed
    assert p["segments"][-1]["k_to"] == pytest.approx([0.25, 0.25, 0.25])
    assert all(float(v).is_integer() for ln in deck.splitlines()[3:-1] for v in ln.split())


def test_edited_label_deck_stays_labels_and_a_coordinate_point_converts(monkeypatch, tmp_path):
    src = _source(INPUTS["si"])
    cfg = {**BASE, **CONFIGS["labels"]}
    computed = kp.parse_band_deck((GOLDEN / "si_labels.d3").read_text())["segments"]
    deck, log, _ = _deck(monkeypatch, tmp_path, src, {**cfg, "band_path": kp.reverse(computed, 1)})
    lines = deck.splitlines()
    assert lines[2].split()[:2] == [str(len(computed)), "0"] and lines[4] == "L G"
    shutil.rmtree(tmp_path / "o")
    deck, log, _ = _deck(monkeypatch, tmp_path, src, {**cfg, "band_path": kp.add_point(computed, "P", [.25, .25, .25])})
    assert deck is not None and deck.splitlines()[2].split()[1] != "0", log[-1500:]


def test_a_bad_edit_writes_no_deck(monkeypatch, tmp_path):
    src = _source(INPUTS["si"])
    deck, log, _ = _deck(monkeypatch, tmp_path, src, {**BASE, **CONFIGS["seekpath"],
                                                      "band_path": [{"from": "GAMMA", "to": "NOPE"}]})
    assert deck is None and "NOPE is not a point of the computed path" in log


def test_config_validation_reports_band_path_shape():
    import d3_config
    ok, errors = d3_config.validate_d3_config({**BASE, "band_path": [{"from": "GAMMA"}]})
    assert not ok and "band_path" in errors[0]
    assert d3_config.validate_d3_config({**BASE, "band_path": kp.from_tokens(GRAPHITE)})[0]


def test_a_title_that_writes_the_jump_like_the_plot_names_every_point():
    """"H|P" in a BAND title (the plot's form of a jump) is the same jump as "-|-": the segments keep their
    labels instead of falling back to K1, K2, ..."""
    import d3_band_path as P
    recs = "\n".join(f"{i} 0 0  {i + 1} 0 0" for i in range(6))
    for path in ("GAMMA-H-N-GAMMA-P-H-|-P-N", "GAMMA-H-N-GAMMA-P-H|P-N"):
        deck = f"BAND\nx - Band Structure - SeeKPath (w.I) - {path}\n6 4 1000 1 10 1 0\n{recs}\nEND"
        d = P.parse_band_deck(deck)
        assert [(s["from"], s["to"]) for s in d["segments"]] == [
            ("GAMMA", "H"), ("H", "N"), ("N", "GAMMA"), ("GAMMA", "P"), ("P", "H"), ("P", "N")], path
