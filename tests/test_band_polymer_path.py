"""A polymer's BAND deck runs along the chain only.

Measured with CRYSTAL23 (STO-3G/PBE LiH chains and (SN)x, SHRINK 8 8):

* A polymer takes no BAND label records. Rod groups 28 (corresponding space
  group P M M M) and 51 (P 4/M M M) got opt2d3's ISS = 0 label deck and
  stopped with "BANDE **** FORMAT ERROR IN INPUT DECK"; a hand-made "G X"
  deck in rod group 1 did too.
* A polymer whose symbol MACE does not map (rod group 4, "P 1 1 21") was read
  as space group 1 and got the triclinic 3D path with I2/I3 != 0. CRYSTAL
  runs it, against manual p.310 note 3 (I2, I3, J2, J3 = 0 in 1D), and puts
  points only where k1 changes: five copies of Gamma-X and nonsense labels.
* "1 2 NSUB ... / 0 0 0  1 0 0" (Gamma -> X = 1/2 along the chain) runs for
  rod groups 1, 4 and 28: "LINE 1 ( 0.00 0.00 0.00: 0.50 0.00 0.00)".

Both real routes run as subprocesses: opt2d3 with Enter or path format 1
(labels) on stdin, and the generator with the workflow's default
configuration. Enter is the label path for a polymer: SeeK-path's (format 4)
ran the chain from Gamma to the next Gamma ("0 0 0  4 0 0" at ISS 4).
"""
import shutil
from pathlib import Path

import pytest

from test_workflow_band_matches_opt2d3 import _manual, _workflow

pytest.importorskip("numpy")

DATA = Path(__file__).parent / "data" / "low_dim_groups"
CHAIN_DECK = ["1 2 1000 1 30 1 0", "0 0 0  1 0 0"]
PARENTS = ["polyyne_rg28", "polyyne_rg51", "polyyne_rg4"]


def _stage(tmp_path, name):
    d = tmp_path / "parent"
    d.mkdir()
    for ext in (".out", ".d12"):
        shutil.copy(DATA / (name + ext), d / (name + ext))
    (d / (name + ".f9")).write_text("wavefunction placeholder\n")
    return d / (name + ".out")


def _chain_records_ok(records):
    for rec in records:
        i1, i2, i3, j1, j2, j3 = (int(x) for x in rec.split())
        assert i2 == i3 == j2 == j3 == 0, rec
        assert i1 != j1, rec


@pytest.mark.parametrize("parent", PARENTS)
def test_label_polymer_deck_is_gamma_x(tmp_path, parent):
    deck = _manual(_stage(tmp_path, parent), tmp_path / "manual", ["1", "1"]).splitlines()
    assert deck[0] == "BAND" and deck[-1] == "END"
    assert deck[2:-1] == CHAIN_DECK, deck
    assert "GAMMA-X" in deck[1], deck[1]


@pytest.mark.parametrize("parent", PARENTS)
def test_enter_through_polymer_deck_is_gamma_x_along_the_chain(tmp_path, parent):
    deck = _manual(_stage(tmp_path, parent), tmp_path / "manual", []).splitlines()
    assert deck[0] == "BAND" and deck[-1] == "END"
    assert deck[2:-1] == CHAIN_DECK, deck
    assert "GAMMA-X" in deck[1], deck[1]


@pytest.mark.parametrize("parent", PARENTS)
def test_workflow_default_polymer_deck_is_the_enter_through_deck(tmp_path, parent):
    from mace.workflow.common.constants import opt2d3_default_band_config
    out = _stage(tmp_path, parent)
    manual = _manual(out, tmp_path / "manual", [])
    assert _workflow(out, tmp_path / "workflow", opt2d3_default_band_config()) == manual


@pytest.mark.parametrize("parent", PARENTS)
@pytest.mark.parametrize("fmt", ["2", "3", "4"])
def test_other_polymer_formats_stay_on_the_chain(tmp_path, parent, fmt):
    deck = _manual(_stage(tmp_path, parent), tmp_path / "manual",
                   ["1", fmt, "", "1", "n"]).splitlines()
    nline = int(deck[2].split()[0])
    assert int(deck[2].split()[1]) > 0, deck
    assert nline == len(deck) - 4 and nline > 0, deck
    _chain_records_ok(deck[3:-1])
