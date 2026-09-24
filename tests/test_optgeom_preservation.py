"""Generated OPT decks keep the parent's OPTGEOM values unless a plan sets them.

* A plan step's ``optimization_settings`` replaced the parent's OPTGEOM
  settings wholesale, so an OPT2 step naming TOLDEG/TOLDEX/TOLDEE/MAXCYCLE
  also dropped the parent's MAXTRADIUS.
* With no plan, a parent OPTGEOM that set no TOLDEG/TOLDEX/TOLDEE (CRYSTAL's
  defaults, 0.0003/0.0012/7) came out with the writer's tight fallbacks
  0.00003/0.00012/7, a tenfold tighter optimisation the parent never ran.
"""
import io

import pytest

from CRYSTALOptToD12 import merge_optimization_settings
from d12_calc_basic import write_optimization_section

import test_numbered_calc_keeps_parent_method as base
from conftest import TEST_DATA

BULK = TEST_DATA / "OPT" / "1_dia_opt_BULK_OPTGEOM"   # OPTGEOM: FULLOPTG, MAXCYCLE 1600 only
PLAN_OPT2 = {"calculation_type": "OPT", "optimization_type": "FULLOPTG",
             "optimization_settings": {"TOLDEG": 0.00003, "TOLDEX": 0.00012,
                                       "TOLDEE": 7, "MAXCYCLE": 800},
             "inherit_settings": True, "customization_level": 0}


def _optgeom(lines):
    i = lines.index("OPTGEOM")
    return [ln.strip() for ln in lines[i + 1:lines.index("ENDOPT", i)]]


# ------------------------------------------------------------------ units

def test_merge_keeps_parent_keys_the_override_does_not_name():
    parent = {"type": "FULLOPTG", "MAXCYCLE": 800, "TOLDEG": 0.0003,
              "TOLDEX": 0.0012, "TOLDEE": 7, "MAXTRADIUS": 0.25}
    merged = merge_optimization_settings(parent, {"TOLDEG": 3e-05, "TOLDEX": 0.00012})
    assert merged == dict(parent, TOLDEG=3e-05, TOLDEX=0.00012)


def test_merge_matches_keys_case_insensitively():
    merged = merge_optimization_settings({"TOLDEG": 0.0003, "MAXTRADIUS": 0.25},
                                         {"toldeg": 3e-05})
    assert merged == {"toldeg": 3e-05, "MAXTRADIUS": 0.25}


def test_merge_drops_the_parent_type_when_the_plan_sets_one():
    merged = merge_optimization_settings({"type": "FULLOPTG", "MAXCYCLE": 800},
                                         {}, replace_type=True)
    assert merged == {"MAXCYCLE": 800}


def test_merge_without_parent_settings():
    assert merge_optimization_settings(None, {"TOLDEE": 8}) == {"TOLDEE": 8}


def _written(settings, **kw):
    f = io.StringIO()
    write_optimization_section(f, "FULLOPTG", settings, **kw)
    return f.getvalue().split()


def test_writer_leaves_unset_tolerances_to_crystal_when_asked():
    assert _written({"type": "FULLOPTG", "MAXCYCLE": 1600},
                    fill_missing_tolerances=False) == [
        "OPTGEOM", "FULLOPTG", "MAXCYCLE", "1600", "ENDOPT"]


def test_writer_still_fills_tolerances_by_default():
    # with the Standard preset (it used to be the Very Tight 0.00003/0.00012)
    assert _written({"MAXCYCLE": 1600})[4:10] == [
        "TOLDEG", "0.0003", "TOLDEX", "0.0012", "TOLDEE", "7"]


# ------------------------------------------------- real decks (corpus-gated)

def _run(tmp_path, stem, steps):
    base._need(stem)
    eng = base._engine(tmp_path, stem.with_suffix(".d12").read_text(),
                       stem.with_suffix(".out").read_text(), steps)
    name, deck = base._only_deck(base._numbered(eng, "OPT2", True))
    return deck.splitlines()


def test_plan_opt2_keeps_the_parents_maxtradius(tmp_path):
    body = _run(tmp_path, base.DIA, {"OPT2_4": PLAN_OPT2})
    assert _optgeom(body) == ["FULLOPTG", "MAXCYCLE", "800", "TOLDEG", "0.00003",
                              "TOLDEX", "0.00012", "TOLDEE", "7",
                              "MAXTRADIUS", "0.25"]


def test_plan_opt2_keeps_parent_values_it_does_not_set(tmp_path):
    step = dict(PLAN_OPT2, optimization_settings={"TOLDEG": 0.0001})
    body = _run(tmp_path, base.DIA, {"OPT2_4": step})
    parent = _optgeom(base.DIA.with_suffix(".d12").read_text().splitlines())
    expected = list(parent)
    expected[expected.index("TOLDEG") + 1] = "0.0001"
    assert _optgeom(body) == expected


def test_no_plan_opt2_adds_no_tolerances_the_parent_never_set(tmp_path):
    body = _run(tmp_path, BULK, None)
    assert _optgeom(body) == ["FULLOPTG", "MAXCYCLE", "1600"]


def test_no_plan_opt2_copies_the_parents_optgeom(tmp_path):
    body = _run(tmp_path, base.DIA, None)
    assert _optgeom(body) == _optgeom(base.DIA.with_suffix(".d12").read_text().splitlines())
