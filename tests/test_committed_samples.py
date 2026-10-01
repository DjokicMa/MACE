"""Corpus sweeps, run over the real CRYSTAL files committed under tests/data.

Several tests sweep every deck or output of the gitignored test/ corpus and so
skip in CI. The checks below are the same checks, run over the real decks and
outputs that ARE committed, so they always run; the corpus sweeps are
unchanged and still run wherever test/ exists.

The committed sample (all real CRYSTAL23 inputs/outputs from HPCC):
  * low_dim_groups/  graphene SLAB SPs in layer groups 37, 47, 80 and carbon
                     chain POLYMER SPs in rod groups 28, 51 (.d12 + .out)
  * opt_restart/     PbTiO3 and Ag2Br3 OPTs with ECP basis sets: the RESTART
                     decks, the walltime-killed first runs and the reruns
  * samples/ecp_decks/  the original PbTiO3 and AgBr OPT decks
"""
import contextlib
import io
import json
import re
from pathlib import Path

import pytest

from d12_parsers import CrystalInputParser
from mace.submission import preflight as pf
from mace.utils.calc_detection import (is_frequency_output, is_optimization_output,
                                       is_transport_output)

from test_deck_geometry_parser import _rebuild, _records, _same_number
from test_preflight_preflight import GOOD_OUT, FakeRunner
from test_title_keyword_detection import _deck_type

DATA = Path(__file__).resolve().parent / "data"
DECKS = sorted(DATA.rglob("*.d12"))
OPT_DECKS = [d for d in DECKS if _deck_type(d) == "OPT"]
EXTERNAL_DECKS = [d for d in DECKS
                  if re.search(r"^\s*99\s+0\s*$", d.read_text(), re.M)
                  and "BASISSET" not in d.read_text()]


def _ids(paths):
    return [p.relative_to(DATA).as_posix() for p in paths]


def test_the_sample_has_every_kind_of_deck_the_sweeps_need():
    kinds = {}
    for d in DECKS:
        with contextlib.redirect_stdout(io.StringIO()):
            kinds.setdefault(CrystalInputParser(str(d)).parse()["dimensionality"], []).append(d)
    counts = {k: len(v) for k, v in kinds.items()}
    assert counts["CRYSTAL"] >= 4 and counts["SLAB"] >= 3 and counts["POLYMER"] >= 2, counts
    assert len(OPT_DECKS) >= 4 and len(EXTERNAL_DECKS) >= 4


# ------------------- test_deck_geometry_parser: every deck rebuilds its geometry

@pytest.mark.parametrize("deck", DECKS, ids=_ids(DECKS))
def test_geometry_rebuilds_from_the_parse(tmp_path, deck):
    with contextlib.redirect_stdout(io.StringIO()):
        parsed = CrystalInputParser(str(deck)).parse()
    assert parsed.get("cell_parameters")
    assert len(parsed["atoms"]) == parsed["n_atoms"]
    orig = _records(deck.read_text().splitlines())
    new = _records(_rebuild(tmp_path, parsed))
    assert len(orig) == len(new)
    for o, n in zip(orig, new):
        assert len(o) == len(n) and all(_same_number(a, b) for a, b in zip(o, n)), (o, n)


@pytest.mark.parametrize("deck", [d for d in DECKS if "graphene" in d.name],
                         ids=lambda d: d.name)
def test_slab_deck_carries_its_layer_group(deck):
    with contextlib.redirect_stdout(io.StringIO()):
        parsed = CrystalInputParser(str(deck)).parse()
    assert parsed["dimensionality"] == "SLAB"
    assert parsed["layer_group"] == parsed["spacegroup"] == int(deck.stem[-2:])


# ------------------- test_external_basis_...: EXTERNAL basis blocks read whole

@pytest.mark.parametrize("deck", EXTERNAL_DECKS, ids=_ids(EXTERNAL_DECKS))
def test_external_deck_round_trips_its_basis_block(deck):
    lines = deck.read_text().split("\n")
    end = next(i for i, l in enumerate(lines) if re.match(r"^\s*99\s+0\s*$", l))
    geo = max(j for j in range(end) if lines[j].strip() == "END")
    want = [l.strip() for l in lines[geo + 1:end] if l.strip()]
    with contextlib.redirect_stdout(io.StringIO()):
        got = CrystalInputParser(str(deck)).parse().get("external_basis_data", [])
    assert got == want
    # the ECP atoms (Z > 200) are part of it
    assert any(re.match(r"^2\d\d\s+\d+$", l) for l in want)


# ------------------- test_preflight_preflight: TESTPDIM on a copy, byte exact

@pytest.mark.parametrize("deck", DECKS, ids=_ids(DECKS))
def test_testpdim_insertion_is_byte_exact_and_before_the_final_end(deck):
    original = pf.read_deck(deck)
    assert "".join(pf.split_lines(original)) == original
    prepared = pf.insert_testpdim(original)
    lines = pf.split_lines(prepared)
    idx = [i for i, ln in enumerate(lines) if ln.strip() == "TESTPDIM"]
    assert len(idx) == 1
    records = [ln.strip() for ln in prepared.splitlines() if ln.strip()]
    assert records[-3:] == ["PPAN", "TESTPDIM", "END"]
    del lines[idx[0]]
    assert "".join(lines) == original


def test_preflight_leaves_the_deck_alone_and_reads_crystals_verdict(tmp_path):
    deck = OPT_DECKS[0]
    before = deck.read_bytes()
    ok = FakeRunner(out=GOOD_OUT)
    result = pf.preflight_deck(deck, runner=ok)
    assert result.status == pf.PASS, result.reason
    assert deck.read_bytes() == before
    assert ok.inputs[0].decode("utf-8", "surrogateescape") == pf.insert_testpdim(pf.read_deck(deck))

    bad = pf.preflight_deck(deck, runner=FakeRunner(
        out=" ERROR **** SGINFO **** WRONG LAYER GROUP\n", returncode=1))
    assert bad.status == pf.FAIL and "SGINFO" in bad.reason
    slow = pf.preflight_deck(deck, runner=FakeRunner(out="", timed_out=True))
    assert slow.status == pf.ERROR and "timed out" in slow.reason
    static = pf.preflight_deck(deck, runner=None)
    assert static.status == pf.SKIPPED and "CRYSTAL was not run" in static.reason


# ------------------- test_title_keyword_detection: outputs typed by their deck

PAIRS = [(o, o.with_suffix(".d12")) for o in sorted(DATA.rglob("*.out"))
         if o.with_suffix(".d12").exists()]


@pytest.mark.parametrize("out, deck", PAIRS, ids=_ids([o for o, _ in PAIRS]))
def test_output_markers_match_the_deck(out, deck):
    content = out.read_text(errors="ignore")
    kind = _deck_type(deck)
    assert is_optimization_output(content) == (kind == "OPT")
    assert not is_frequency_output(content)
    assert not is_transport_output(content)


@pytest.mark.parametrize("deck", OPT_DECKS, ids=_ids(OPT_DECKS))
def test_formula_update_treats_real_opt_decks_as_opt(deck):
    from mace.utils.formula_extractor import _updates_formula_as_opt_deck
    assert _updates_formula_as_opt_deck(deck, deck.read_text(errors="ignore"))


# ------------------- test_settings_extractor: a real deck into the database

def test_input_settings_of_a_real_deck_reach_the_database(tmp_path):
    from mace.database.materials import MaterialDatabase
    from mace.utils.settings_extractor import extract_and_store_input_settings

    deck = DATA / "opt_restart" / "tqb_pto.d12"
    db_path = str(tmp_path / "m.db")
    db = MaterialDatabase(db_path=db_path, ase_db_path=str(tmp_path / "s.db"))
    db.create_material(material_id="tqb_pto", formula="PbTiO3")
    calc_id = db.create_calculation("tqb_pto", "OPT", input_file=str(deck))
    with contextlib.redirect_stdout(io.StringIO()):
        assert extract_and_store_input_settings(calc_id, deck, db_path) is True
    with db._get_connection() as conn:
        row = conn.execute("SELECT input_settings_json FROM calculations WHERE calc_id = ?",
                           (calc_id,)).fetchone()
    settings = json.loads(row[0])
    assert "crystal_keywords" in settings


# ------------------- test_aggregation_keys: the extractor emits what's grouped by

def test_aggregation_keys_from_a_real_sp_output(tmp_path):
    from mace.utils.property_extractor import CrystalPropertyExtractor
    from test_aggregation_keys import GROUPING_KEYS
    ex = CrystalPropertyExtractor(db_path=str(tmp_path / "agg.db"), enable_tracking=False)
    with contextlib.redirect_stdout(io.StringIO()):
        props = ex.extract_all_properties(DATA / "low_dim_groups" / "graphene_lg37.out",
                                          material_id="m", calc_id="c")
    for key in GROUPING_KEYS:
        assert props.get(key) is not None, key
    assert props["total_energy_au"] == pytest.approx(-151.85643847112)
    assert props["atoms_in_unit_cell"] == 4
