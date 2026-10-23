"""Calculation Setup shows a parent's own SCF tolerances as the selected level.

phonopy's Si example runs TOLINTEG 8 8 8 8 16 / TOLDEE 9, none of the presets.
The form's tolerance select had only Standard/Tight/Very tight/Custom and the
raw "8 8 8 8 16" outside them; it now has opt2d12's own "Keep the parent's"
option (its menu line, read from the CLI's transcript) selected, with its
values, and picking a preset or the parent's level again writes that deck.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

import guiharness as H

sys.path.insert(0, str(H.REPO_ROOT))
from mace.gui import setup as S  # noqa: E402

SI = H.REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "Si"
KEEP = "Keep the parent's"


def _cache():
    store = {}
    return (lambda k: store.get(json.dumps(k, default=str)),
            lambda k, v, n: store.__setitem__(json.dumps(k, default=str), v))


def _run(tmp, src, calc, answers=None, cache=None):
    get, put = cache or _cache()
    req = S.Request({"answers": answers or {}, "patches": {}, "preview": True, "json_text": None})
    work = tmp / f"w{len(list(tmp.glob('w*')))}"
    return S.run_setup(req, "opt2d12", calc, work, src, src.name, S.source_key("opt2d12", src), get, put, None,
                       schema_info=S.schema())


def _row(r, fid):
    return next(row for row in r["form"] if row["id"] == fid)


def _tol(deck):
    lines = [x.strip() for x in deck.splitlines()]
    at = lines.index("TOLINTEG")
    return " ".join(lines[at + 1].split()), lines[lines.index("TOLDEE", at) + 1]


@pytest.fixture
def si(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name in ("parent.out", "parent.d12"):
        shutil.copy2(SI / name, src / name)
    return src / "parent.out"


def test_tolerance_select_defaults_to_the_parents_values(tmp_path, si):
    cache = _cache()
    r = _run(tmp_path, si, "SP", cache=cache)
    row = _row(r, "tolerance")
    heads = [o["head"] for o in row["options"]]
    assert heads == ["Standard", "Tight", "Very tight", "Custom", KEEP]
    assert row["value"] == KEEP and row["default"] == KEEP
    keep = row["options"][-1]
    assert keep["num"] == "5" and keep["values"] == "TOLINTEG: 8 8 8 8 16, TOLDEE: 9"
    assert json.loads(r["json_text"])["tolerances"] == {"TOLINTEG": "8 8 8 8 16", "TOLDEE": 9}
    assert _tol(r["deck"]) == ("8 8 8 8 16", "9")

    tight = _run(tmp_path, si, "SP", {"tolerance": "Tight"}, cache=cache)
    assert _row(tight, "tolerance")["value"] == "Tight"
    assert _tol(tight["deck"]) == ("8 8 8 9 24", "9")

    back = _run(tmp_path, si, "SP", {"tolerance": KEEP}, cache=cache)
    assert _row(back, "tolerance")["value"] == KEEP
    assert _tol(back["deck"]) == ("8 8 8 8 16", "9")


@pytest.mark.parametrize("tol, head, want", [
    (("10 10 10 12 40", "12"), KEEP, ("10 10 10 12 40", "12")),
    (("10 8 10 8 50", "9"), "Very tight, or the parent's where tighter", ("10 9 10 11 50", "11")),
])
def test_freq_select_is_never_looser_than_the_parent(tmp_path, tol, head, want):
    """FREQ's Very tight default keeps each value a parent sets tighter (opt2d12's rule)."""
    src = tmp_path / "src"
    src.mkdir()
    text = (SI / "parent.d12").read_text()
    text = text.replace("TOLINTEG\n8 8 8 8 16", f"TOLINTEG\n{tol[0]}").replace("TOLDEE\n9", f"TOLDEE\n{tol[1]}")
    (src / "si.d12").write_text(text)
    r = _run(tmp_path, src / "si.d12", "FREQ")
    row = _row(r, "tolerance")
    assert row["value"] == head and row["default"] == head
    assert next(o for o in row["options"] if o["head"] == head)["values"] == f"TOLINTEG: {want[0]}, TOLDEE: {want[1]}"
    assert _tol(r["deck"]) == want
