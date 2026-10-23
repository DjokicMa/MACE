"""A chosen SeeK-path BAND path (format 4) of a slab or polymer is the zone's own path.

78aba155 made the label path the default for slabs and polymers but left an
explicit format 4 as it was: a slab in a non-centrosymmetric layer group got
the primed time-reversal copy of every segment from the static no-inversion
tables (MgO slab, layer group 1: GAMMA-X-|-Y-GAMMA-V-|-GAMMA-X'-|-Y'-GAMMA-V',
six segments for three), and a polymer ran from Gamma to the next Gamma
("0 0 0  4 0 0" at ISS 4: SeeK-path's primitive cell of the 3D box is half
the chain's cell when its atoms are evenly spaced). With time reversal (no
two-component SCF) E(k) = E(-k) and the primed half goes; a two-component
parent keeps it. A polymer's path ends at X = 1/2. 3D decks are unchanged
(checked byte for byte on the 3D corpus parents against 1f5d713f).

Real outputs (tests/data/orbitals/scf/mgoslab.out, tests/data/low_dim_groups,
the corpus's four-layer graphene slabs); opt2d3 and the generator with a
format-4 configuration run as subprocesses.
"""
import shutil
from pathlib import Path

import pytest

from conftest import TEST_DATA
from test_workflow_band_matches_opt2d3 import _manual, _workflow

pytest.importorskip("numpy")

DATA = Path(__file__).parent / "data"
SLAB_DECK = ("SeeKPath (no.I) - GAMMA-X-|-Y-GAMMA-V", ["3 6 1000 1 58 1 0", "0 0 0  3 0 0", "0 3 0  0 0 0",
                                                       "0 0 0  3 3 0"])
CHAIN_DECK = ["1 2 1000 1 30 1 0", "0 0 0  1 0 0"]


def _stage(tmp_path, src, deck_text=None):
    d = tmp_path / "parent"
    d.mkdir()
    shutil.copy(src, d / src.name)
    if deck_text is not None:
        (d / src.with_suffix(".d12").name).write_text(deck_text)
    elif src.with_suffix(".d12").exists():
        shutil.copy(src.with_suffix(".d12"), d / src.with_suffix(".d12").name)
    (d / src.with_suffix(".f9").name).write_text("wavefunction placeholder\n")
    return d / src.name


def _seekpath_config():
    from mace.workflow.common.constants import opt2d3_default_band_config
    return {k: v for k, v in opt2d3_default_band_config().items() if k != "kpath_default"}


def test_slab_without_inversion_has_no_primed_copy(tmp_path):
    out = _stage(tmp_path, DATA / "orbitals" / "scf" / "mgoslab.out")
    deck = _manual(out, tmp_path / "manual", ["1", "4"]).splitlines()
    assert deck[1].endswith(SLAB_DECK[0]), deck[1]
    assert deck[2:-1] == SLAB_DECK[1], deck
    assert _workflow(out, tmp_path / "workflow", _seekpath_config()).splitlines() == deck


def test_two_component_slab_keeps_the_time_reversal_copy(tmp_path):
    """A TWOCOMPON parent (here the deck beside the real output says so): -k is not k."""
    out = _stage(tmp_path, DATA / "orbitals" / "scf" / "mgoslab.out",
                 "mgoslab\nSLAB\n1\n4.2 4.2 90\n1\n12 0 0 0\nTWOCOMPON\nSOC\nEND\nEND\n")
    deck = _manual(out, tmp_path / "manual", ["1", "4"]).splitlines()
    assert deck[1].endswith("GAMMA-X-|-Y-GAMMA-V-|-GAMMA-X'-|-Y'-GAMMA-V'"), deck[1]
    assert int(deck[2].split()[0]) == 6


@pytest.mark.parametrize("parent", ["polyyne_rg4", "polyyne_rg28", "polyyne_rg51"])
def test_polymer_path_ends_at_x(tmp_path, parent):
    out = _stage(tmp_path, DATA / "low_dim_groups" / f"{parent}.out")
    deck = _manual(out, tmp_path / "manual", ["1", "4"]).splitlines()
    assert "SeeKPath" in deck[1] and deck[1].endswith("GAMMA-X"), deck[1]
    assert deck[2:-1] == CHAIN_DECK, deck
    assert _workflow(out, tmp_path / "workflow", _seekpath_config()).splitlines() == deck


def test_centrosymmetric_slab_is_unchanged(tmp_path):
    out = _stage(tmp_path, DATA / "low_dim_groups" / "graphene_lg80.out")
    deck = _manual(out, tmp_path / "manual", ["1", "4"]).splitlines()
    assert deck[1].endswith("SeeKPath (w.I) - GAMMA-M-K-GAMMA")
    assert deck[2:-1] == ["3 12 1000 1 30 1 0", "0 0 0  6 0 0", "6 0 0  4 4 0", "4 4 0  0 0 0"]


def test_corpus_slab_without_inversion(tmp_path):
    if not TEST_DATA.is_dir():
        pytest.skip("test/ corpus not present")
    src = next(iter(sorted(TEST_DATA.rglob("4LG_2x2_AA_opt_HSESOL3C_optimized.out"))), None)
    if src is None:
        pytest.skip("four-layer graphene slab not in this corpus")
    deck = _manual(_stage(tmp_path, src), tmp_path / "manual", ["1", "4"]).splitlines()
    assert deck[1].endswith("SeeKPath (no.I) - GAMMA-X-|-Y-GAMMA-V"), deck[1]
    assert deck[3:-1] == ["0 0 0  3 0 0", "0 3 0  0 0 0", "0 0 0  3 3 0"], deck


def test_time_reversal_half_keeps_a_path_without_copies():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Crystal_d3"))
    from d3_kpoints import chain_path_in_zone, time_reversal_half
    path = [[0, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 0.5, 0.5, 0]]
    assert time_reversal_half(path, ["GAMMA", "X", "S"]) == (path, ["GAMMA", "X", "S"])
    assert time_reversal_half(path + [[0, 0, 0, -0.5, 0, 0]], ["GAMMA", "X", "S", "|", "GAMMA", "X'"]) == \
        (path, ["GAMMA", "X", "S"])
    assert chain_path_in_zone([[0, 0, 0, 2, 0, 0]], 4) == [[0, 0, 0, 2, 0, 0]]
    assert chain_path_in_zone([[0, 0, 0, 4, 0, 0]], 4) == []
