"""The default BAND path is SeeK-path for a 3D crystal and the labels for a slab or polymer.

1.1.6 made SeeK-path (format 4) opt2d3's Enter and the workflow's default.
For a slab or polymer that path is the corresponding 3D group's, cut to the
plane or chain: a slab in a non-centrosymmetric layer group got the primed
time-reversal copy of every segment (MgO slab, layer group 1: six segments,
GAMMA-X-|-Y-GAMMA-V-|-GAMMA-X'-|-Y'-GAMMA-V'), and a polymer ran its chain
from Gamma to the next Gamma ("0 0 0  4 0 0" at ISS 4, not Gamma -> X). The
default for them is the label path again (format 1), the deck 1.1.5 wrote
(checked byte for byte against 1.1.6-preview's decks of the same parents);
a chosen format 4 is kept, and a 3D crystal keeps SeeK-path.

Real outputs (tests/data/low_dim_groups, orbitals/scf/mgoslab.out,
opt_restart/tqb_pto.out); both routes run as subprocesses: opt2d3 with piped
answers and the generator with the workflow's default configuration.
"""
import shutil
from pathlib import Path

import pytest

from test_workflow_band_matches_opt2d3 import _manual, _workflow

pytest.importorskip("numpy")

DATA = Path(__file__).parent / "data"
# the 1.1.5 / 1.1.6-preview label decks (format 1) of these parents
LABEL_DECKS = {
    "mgoslab": ("default - V-Y-GAMMA-X", ["3 6 1000 1 58 1 0", "3 3 0  0 3 0", "0 3 0  0 0 0", "0 0 0  3 0 0"]),
    "graphene_lg80": ("default - M-GAMMA-K", ["2 12 1000 1 30 1 0", "6 0 0  0 0 0", "0 0 0  4 4 0"]),
    "polyyne_rg4": ("default - GAMMA-X", ["1 2 1000 1 30 1 0", "0 0 0  1 0 0"]),
}
SOURCES = {"mgoslab": DATA / "orbitals" / "scf" / "mgoslab.out",
           "graphene_lg80": DATA / "low_dim_groups" / "graphene_lg80.out",
           "polyyne_rg4": DATA / "low_dim_groups" / "polyyne_rg4.out",
           "tqb_pto": DATA / "opt_restart" / "tqb_pto.out"}


def _stage(tmp_path, name):
    d = tmp_path / "parent"
    d.mkdir()
    src = SOURCES[name]
    shutil.copy(src, d / f"{name}.out")
    if src.with_suffix(".d12").exists():
        shutil.copy(src.with_suffix(".d12"), d / f"{name}.d12")
    (d / f"{name}.f9").write_text("wavefunction placeholder\n")
    return d / f"{name}.out"


@pytest.mark.parametrize("name", sorted(LABEL_DECKS))
def test_enter_gives_a_slab_or_polymer_the_label_path(tmp_path, name):
    out = _stage(tmp_path, name)
    deck = _manual(out, tmp_path / "manual", []).splitlines()
    title, records = LABEL_DECKS[name]
    assert deck[1].endswith(title), deck[1]
    assert deck[2:-1] == records
    assert _manual(out, tmp_path / "labels", ["1", "1"]).splitlines() == deck     # = format 1


@pytest.mark.parametrize("name", sorted(LABEL_DECKS))
def test_the_workflow_default_is_the_same_deck(tmp_path, name):
    from mace.workflow.common.constants import opt2d3_default_band_config
    out = _stage(tmp_path, name)
    assert _workflow(out, tmp_path / "workflow", opt2d3_default_band_config()) == \
        _manual(out, tmp_path / "manual", [])


def test_a_chosen_seekpath_path_is_kept(tmp_path):
    """Format 4 typed at opt2d3, or a plan's SeeK-path configuration that is
    not the default (an explicit choice, or a plan made before this release)."""
    from mace.workflow.common.constants import opt2d3_default_band_config
    out = _stage(tmp_path, "mgoslab")
    deck = _manual(out, tmp_path / "manual", ["1", "4"]).splitlines()
    assert "SeeKPath" in deck[1] and int(deck[2].split()[0]) == 3      # no primed copy (test_band_format4_low_dim)
    chosen = {k: v for k, v in opt2d3_default_band_config().items() if k != "kpath_default"}
    assert _workflow(out, tmp_path / "workflow", chosen).splitlines() == deck


def test_a_3d_crystal_keeps_seekpath(tmp_path):
    from mace.workflow.common.constants import opt2d3_default_band_config
    out = _stage(tmp_path, "tqb_pto")
    deck = _manual(out, tmp_path / "manual", [])
    assert "SeeKPath" in deck.splitlines()[1]
    assert _workflow(out, tmp_path / "workflow", opt2d3_default_band_config()) == deck


def test_the_planners_enter_is_the_default(tmp_path, monkeypatch):
    """Advanced BAND: path option 1 (automatic), Enter at the format."""
    from test_workflow_band_matches_opt2d3 import _advanced
    cfg = _advanced(tmp_path, monkeypatch, ["1", "", ""])
    assert cfg["seekpath_full"] is True and cfg["kpath_default"] is True
    cfg = _advanced(tmp_path / "b", monkeypatch, ["1", "4", ""])
    assert cfg["seekpath_full"] is True and "kpath_default" not in cfg
