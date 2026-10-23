"""The default BAND deck is one CRYSTAL23 can read, for every lattice.

opt2d3's high-symmetry label format (path format 1, no longer the Enter-through
answer: SeeK-path is) writes ISS = 0 and label records like "V Y". CRYSTAL23
takes those labels only from its Tables 14.1/14.2 (manual note 6: only the
labels in those tables are recognized), and only for 3D systems:

* Triclinic (SG 1-2) has no table entry. A P-1 LiH label deck stopped in
  Pproperties with "BANDE **** K-POINTS NOT CODIFIED", a minimal "G X" deck
  too; the same path written as integer coordinates ran.
* A slab takes no label input at all, even in a tabulated lattice: hBN in
  layer groups 1 and 78 (SG 187) stopped with "BANDE **** FORMAT ERROR IN
  INPUT DECK"; the integer deck of the same in-plane points ran.

Those parents now get the deck opt2d3's k-point vector format (2) writes:
the same path as integer coordinates in units of 1/ISS, ISS from the
parent's shrink. 3D lattices with a table entry keep their label deck (an
fcc MgO "X G / G L / L W / W G" deck runs).

Both real routes run as subprocesses: opt2d3 with path format 1 on stdin, and
the generator with the matching label configuration.
"""
import shutil
from pathlib import Path

import pytest

from test_workflow_band_matches_opt2d3 import _manual, _workflow

pytest.importorskip("numpy")

DATA = Path(__file__).parent / "data"
ENTER = ["1", "1"]                  # automatic path, format 1: the high-symmetry labels
FORMAT_2 = ["1", "2", "", "1", "n"]
# the workflow's label configuration (the engine default before SeeK-path became the default)
LABELS_CONFIG = {"calculation_type": "BAND", "path_method": "labels", "path": "auto", "labels": "auto",
                 "auto_path": True, "shrink": 0, "n_points": 1000, "bands": "auto", "auto_bands": True,
                 "first_band": 1, "last_band": None, "plot": True, "print_eigenvalues": False}

EXPECTED = {
    # textbook LiH, rocksalt primitive cell in P -1 (synthetic excerpt)
    "lih_p1bar": (DATA / "label_band" / "lih_p1bar.out", [
        "9 6 1000 1 6 1 0",
        "3 3 0  0 3 0", "0 3 0  0 0 0", "0 0 0  0 0 3", "0 0 3  0 3 3",
        "0 3 3  3 3 3", "3 3 3  0 0 0", "0 0 0  3 0 0", "3 0 0  3 0 3",
        "3 0 3  0 0 0"]),
    # textbook hBN monolayer in layer group 1 (synthetic excerpt)
    "hbn_p1_slab": (DATA / "label_band" / "hbn_p1_slab.out", [
        "3 8 1000 1 10 1 0",
        "4 4 0  0 4 0", "0 4 0  0 0 0", "0 0 0  4 0 0"]),
    # graphene, layer group 80 (SG 191): a slab in a tabulated lattice
    "graphene_lg80": (DATA / "low_dim_groups" / "graphene_lg80.out", [
        "2 12 1000 1 30 1 0",
        "6 0 0  0 0 0", "0 0 0  4 4 0"]),
}


def _stage(tmp_path, src):
    d = tmp_path / "parent"
    d.mkdir()
    for ext in (".out", ".d12"):
        if src.with_suffix(ext).exists():
            shutil.copy(src.with_suffix(ext), d / src.with_suffix(ext).name)
    (d / src.with_suffix(".f9").name).write_text("wavefunction placeholder\n")
    return d / src.name


@pytest.mark.parametrize("parent", sorted(EXPECTED))
def test_enter_through_deck_is_integer_coordinates(tmp_path, parent):
    src, records = EXPECTED[parent]
    out = _stage(tmp_path, src)
    deck = _manual(out, tmp_path / "manual", ENTER).splitlines()
    assert deck[0] == "BAND" and deck[-1] == "END"
    assert deck[2:-1] == records
    # ... which is the deck opt2d3's k-point vector format writes
    assert _manual(out, tmp_path / "format2", FORMAT_2).splitlines() == deck


@pytest.mark.parametrize("parent", sorted(EXPECTED))
def test_workflow_label_deck_is_the_opt2d3_label_deck(tmp_path, parent):
    src, records = EXPECTED[parent]
    out = _stage(tmp_path, src)
    manual = _manual(out, tmp_path / "manual", ENTER)
    assert _workflow(out, tmp_path / "workflow", dict(LABELS_CONFIG)) == manual
    assert manual.splitlines()[2:-1] == records


# Corpus parents of the same two kinds (skipped without test/)
CORPUS = {
    "OPT/Ti7O13_mp-556724_sg2_*_optimized.out": [   # P -1
        "9 8 1000 1 1028 1 0",
        "4 4 0  0 4 0", "0 4 0  0 0 0", "0 0 0  0 0 4", "0 0 4  0 4 4",
        "0 4 4  4 4 4", "4 4 4  0 0 0", "0 0 0  4 0 0", "4 0 0  4 0 4",
        "4 0 4  0 0 0"],
    "OPT/4LG_2x2_ABAB_opt_HSESOL3C_optimized.out": [  # slab, plane group 1
        "3 6 1000 1 1440 1 0",
        "3 3 0  0 3 0", "0 3 0  0 0 0", "0 0 0  3 0 0"],
}


@pytest.mark.parametrize("pattern", sorted(CORPUS))
def test_corpus_enter_through_deck_is_integer_coordinates(tmp_path, pattern):
    from conftest import find_data
    out = _stage(tmp_path, find_data(pattern))
    manual = _manual(out, tmp_path / "manual", ENTER)
    assert manual.splitlines()[2:-1] == CORPUS[pattern]
    assert _manual(out, tmp_path / "format2", FORMAT_2) == manual
    assert _workflow(out, tmp_path / "workflow", dict(LABELS_CONFIG)) == manual


def test_tabulated_3d_lattice_keeps_its_label_deck(tmp_path):
    out = _stage(tmp_path, DATA / "opt_restart" / "tqb_pto.out")
    deck = _manual(out, tmp_path / "manual", ENTER).splitlines()
    assert deck[2].split()[1] == "0"
    assert all(rec.replace(" ", "").isalpha() for rec in deck[3:-1]), deck


def test_label_with_no_coordinates_writes_no_deck(tmp_path, monkeypatch):
    """A label CRYSTAL23 does not know for the lattice (W is fcc, the parent
    is simple cubic) and MACE has no coordinates for: no deck, rather than
    the old placeholder records '0.000000 0.000000 0.000000' with ISS 0."""
    import CRYSTALOptToD3 as d3gen
    out = _stage(tmp_path, DATA / "opt_restart" / "tqb_pto.out")
    monkeypatch.setattr(d3gen.D3Generator, "_copy_wavefunction", lambda self: True)
    (tmp_path / "o").mkdir()
    gen = d3gen.D3Generator(str(out), "BAND", output_dir=str(tmp_path / "o"))
    assert gen.generate_d3(shared_config={"path_method": "labels", "shrink": 0,
                                          "path": ["GAMMA", "X", "W"]}) is None
    assert not list((tmp_path / "o").glob("*.d3"))


def test_triclinic_labels_are_not_crystal23_labels():
    """The FREQ phonon-band writer asks the same question and turns an
    invalid label path into coordinates."""
    from d3_kpoints import validate_kpoint_labels_for_crystal23, get_band_path_from_symmetry
    for sg in (1, 2):
        path = get_band_path_from_symmetry(sg, "P")
        assert validate_kpoint_labels_for_crystal23(path, sg, "P")[0] is False
    assert validate_kpoint_labels_for_crystal23(["X", "GAMMA", "L", "W"], 225, "F")[0] is True
