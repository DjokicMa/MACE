"""Edited band k-paths that CRYSTAL23 could not run.

* slabs and polymers: the non-periodic k components are formally zero
  (CRYSTAL23 manual p. 309, BAND note 3), so an added point with k_z (slab)
  or k_y/k_z (polymer) is refused, in the opt2d3 prompt and in a band_path;
* a labels-only edit: CRYSTAL23 reads only the Tables 14.1/14.2 labels
  (p. 309, note 6, else "K-POINTS NOT CODIFIED"); monoclinic Ag2S (sg 11)
  with Z-Q or GAMMA-X is refused by name, and a label with coordinates is
  written as coordinates.

Parents: real outputs (graphene slab, polyyne, Ag2S, Si).
"""
import contextlib
import copy
import io
import shutil
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
import d3_band_path as kp  # noqa: E402

SLAB = REPO_ROOT / "examples" / "gui" / "outputs" / "graphene_slab_sp.out"
POLY = REPO_ROOT / "tests" / "data" / "low_dim_groups" / "polyyne_rg4.out"
AG2S = REPO_ROOT / "tests" / "data" / "label_band" / "ag2s_sg11_sp.out"
SI = REPO_ROOT / "examples" / "gui" / "outputs" / "Si_sp.out"
BASE = {"calculation_type": "BAND", "n_points": 1000, "bands": "auto", "auto_bands": True,
        "first_band": 1, "last_band": None, "plot": True, "print_eigenvalues": False}
COORDS = {"path_method": "coordinates", "path": "auto", "auto_path": True, "shrink": "auto"}
LABELS = {"path_method": "labels", "path": "auto", "auto_path": True, "shrink": 0}


def _deck(monkeypatch, tmp_path, src, cfg):
    import CRYSTALOptToD3 as d3gen
    monkeypatch.setattr(d3gen.D3Generator, "_copy_wavefunction", lambda self: True)
    work = tmp_path / "w"
    work.mkdir(parents=True, exist_ok=True)
    for f in src.parent.glob(src.stem + ".*"):
        shutil.copy(f, work / f.name)
    out = tmp_path / "o"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        g = d3gen.D3Generator(str(work / src.name), "BAND", output_dir=str(out))
        g.generate_d3(copy.deepcopy(cfg))
    decks = sorted(out.glob("*.d3")) if out.exists() else []
    return (decks[0].read_text() if decks else None), buf.getvalue(), g


# ---------------------------------------------------------------- note 3: non-periodic k = 0
def test_validate_refuses_out_of_plane_points():
    segs = kp.from_tokens(["GAMMA", "M"], [[0, 0, 0, .5, 0, 0]])
    slab = kp.add_point(segs, "Q", [.25, .25, .5])
    err, _ = kp.validate(slab, segs, dimensionality=2)
    assert err and "k_z = 0.5" in err[0] and "p. 309" in err[0]
    assert kp.validate(slab, segs, dimensionality=3)[0] == []
    assert kp.validate(kp.add_point(segs, "Q", [.25, .25, 0]), segs, dimensionality=2)[0] == []
    poly = kp.add_point(segs, "P", [.25, .25, 0])
    err, _ = kp.validate(poly, segs, dimensionality=1)
    assert err and "k_y = 0.25" in err[0] and "polymer" in err[0]
    assert kp.validate(kp.add_point(segs, "P", [.25, 0, 0]), segs, dimensionality=1)[0] == []


def test_the_prompt_refuses_an_out_of_plane_point():
    computed = kp.from_tokens(["GAMMA", "M"], [[0, 0, 0, .5, 0, 0]])
    answers = iter(["a Q 0.25 0.25 0.5", "a Q 0.25 0.25 0", ""])
    said = []
    got = kp.edit_interactively(computed, lambda _: next(answers), said.append, dimensionality=2)
    assert any("k_z = 0.5" in s for s in said)
    assert got[-1]["k_to"] == [0.25, 0.25, 0.0]


@pytest.mark.parametrize("src,k,ok", [(SLAB, [.25, .25, .5], False), (SLAB, [.25, .25, 0], True),
                                      (POLY, [.25, .25, 0], False), (POLY, [.25, 0, 0], True)])
def test_band_path_decks_of_slabs_and_polymers(monkeypatch, tmp_path, src, k, ok):
    plain, log, _ = _deck(monkeypatch, tmp_path / "c", src, {**BASE, **COORDS})
    assert plain is not None, log[-1500:]
    computed = kp.parse_band_deck(plain)["segments"]
    edit = kp.add_point(computed, "Q", k)
    deck, log, g = _deck(monkeypatch, tmp_path, src, {**BASE, **COORDS, "band_path": edit})
    dim = g.structure_info.get("dimensionality")
    assert dim == (2 if src is SLAB else 1)
    if not ok:
        assert deck is None and "p. 309" in log, log[-1500:]
        return
    assert deck is not None, log[-1500:]
    for line in deck.splitlines()[3:-1]:
        v = [int(x) for x in line.split()]
        assert all(v[i] == 0 and v[3 + i] == 0 for i in range(dim, 3)), line


# ---------------------------------------------------------------- note 6: labels CRYSTAL reads
@pytest.mark.parametrize("edit,label", [([{"from": "Z", "to": "Q"}], "Q"),
                                        ([{"from": "GAMMA", "to": "X"}], "X")])
def test_monoclinic_labels_crystal_cannot_read_are_refused(monkeypatch, tmp_path, edit, label):
    deck, log, g = _deck(monkeypatch, tmp_path, AG2S, {**BASE, **LABELS, "band_path": edit})
    assert g.structure_info.get("space_group") == 11
    assert deck is None, deck
    assert f"names {label}" in log and "K-POINTS NOT CODIFIED" in log


def test_readable_monoclinic_labels_stay_labels(monkeypatch, tmp_path):
    deck, log, _ = _deck(monkeypatch, tmp_path, AG2S, {**BASE, **LABELS,
                                                         "band_path": [{"from": "Z", "to": "GAMMA"},
                                                                       {"from": "GAMMA", "to": "Y"}]})
    assert deck is not None, log[-1500:]
    lines = deck.splitlines()
    assert lines[2].split()[1] == "0" and lines[3:5] == ["Z G", "G Y"]


def test_a_label_with_coordinates_is_written_as_coordinates(monkeypatch, tmp_path):
    """A label in MACE's table that CRYSTAL does not read (as if L were not
    in Table 14.1): every point of the edit gets its coordinates, with a note."""
    import CRYSTALOptToD3
    # the d3_kpoints module opt2d3 imported (as Crystal_d3.d3_kpoints or d3_kpoints)
    d3_kpoints = sys.modules[CRYSTALOptToD3.validate_kpoint_labels_for_crystal23.__module__]
    _, _, g = _deck(monkeypatch, tmp_path, SI, {**BASE, **LABELS})
    system = d3_kpoints.get_crystal_system_from_space_group(227, "F")
    edit = [{"from": "L", "to": "GAMMA"}, {"from": "GAMMA", "to": "X"}]
    assert g._crystal_readable_label_edit(edit) == edit
    monkeypatch.setitem(d3_kpoints.CRYSTAL23_VALID_LABELS, system,
                        d3_kpoints.CRYSTAL23_VALID_LABELS[system] - {"L"})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        got = g._crystal_readable_label_edit(edit)
    assert "written as coordinates" in buf.getvalue()
    table = d3_kpoints.KPOINT_COORDINATES[system]
    assert got[0]["k_from"] == table["L"] and got[1]["k_to"] == table["X"]


def test_gui_setup_reads_the_parent_dimensionality():
    from mace.gui import setup
    assert [setup._kpath_dimensionality(p) for p in (SLAB, POLY, AG2S, None)] == [2, 1, 3, 3]
