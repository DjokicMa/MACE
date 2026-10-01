"""A slab's BAND deck explores the plane of the slab only.

CRYSTAL23 manual p.310, BAND note 3: "in two-(one-) dimensional cases I3, J3
(I2,I3,J2,J3) are formally input as zero". A SLAB output says
"SLAB CALCULATION" and names its layer group and the corresponding space
group ("TWO-SIDED PLANE GROUP N. 80 : P 6/M M M", "CORRESPONDING SPACE GROUP
N. 191"; App. A.2 p.421: layer groups are identified by the corresponding
space group). The outputs used are real CRYSTAL23 slab outputs of graphene in
layer groups 37 (pmmm), 47 (cmmm) and 80 (p6/mmm).
"""
import shutil
from pathlib import Path

import pytest

import CRYSTALOptToD3 as d3gen

DATA = Path(__file__).parent / "data" / "low_dim_groups"


@pytest.mark.parametrize("name,space_group,lattice", [
    ("graphene_lg37", 47, "P"),
    ("graphene_lg47", 65, "C"),
    ("graphene_lg80", 191, "P"),
])
def test_slab_symmetry_is_read(name, space_group, lattice):
    info = d3gen.D3Generator(str(DATA / f"{name}.out"), "BAND").structure_info
    assert info["dimensionality"] == 2
    assert info["space_group"] == space_group
    assert info["lattice_type"] == lattice


def _band_deck(tmp_path, monkeypatch, name, **config):
    for ext in (".out", ".d12"):
        shutil.copy(DATA / f"{name}{ext}", tmp_path)
    monkeypatch.setattr(d3gen.D3Generator, "_copy_wavefunction", lambda self: True)
    gen = d3gen.D3Generator(str(tmp_path / f"{name}.out"), "BAND", output_dir=str(tmp_path))
    gen.generate_d3(shared_config=dict(config, auto_path=True, n_points=1000))
    lines = (tmp_path / f"{name}_band.d3").read_text().splitlines()
    assert lines[0] == "BAND" and lines[-1] == "END"
    nline, iss = (int(v) for v in lines[2].split()[:2])
    records = lines[3:-1]
    assert len(records) == nline > 0
    return lines[1], iss, records


ROUTES = {
    "seekpath": dict(path_method="coordinates", seekpath_full=True),
    "literature": dict(path_method="coordinates", literature_path=True),
    "vectors": dict(path_method="coordinates"),
    "labels": dict(path_method="labels", path="auto"),
}


def _title_edges(title):
    path = title.rpartition(" - ")[2].split("-")
    return [(a, b) for a, b in zip(path, path[1:]) if "|" not in (a, b)]


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("name", ["graphene_lg37", "graphene_lg47", "graphene_lg80"])
def test_slab_path_stays_in_plane(tmp_path, monkeypatch, name, route):
    title, iss, records = _band_deck(tmp_path, monkeypatch, name, **ROUTES[route])
    if iss == 0:
        # label mode: no label off the kz = 0 plane (Table 14.1: A, L, H of
        # the hexagonal lattice; Table 14.2: Z, U, R, T of the orthorhombic)
        off_plane = {"A", "L", "H", "Z", "U", "R", "T"}
        assert not {lab for rec in records for lab in rec.split()} & off_plane, records
    else:
        for rec in records:
            ints = [int(v) for v in rec.split()]
            assert ints[2] == 0 and ints[5] == 0, rec
        if "in-plane path" not in title:
            assert len(_title_edges(title)) == len(records), title


def test_hexagonal_slab_seekpath_is_gamma_m_k_gamma(tmp_path, monkeypatch):
    """Graphene (layer group 80, P6/mmm), with or without the seekpath
    library: SeeK-path's hP2 path cut to the plane, K = (1/3, 1/3, 0) exact."""
    title, iss, records = _band_deck(tmp_path, monkeypatch, "graphene_lg80",
                                     **ROUTES["seekpath"])
    assert title.endswith(" - GAMMA-M-K-GAMMA")
    assert iss % 6 == 0
    k = iss // 3
    assert records == [f"0 0 0  {iss // 2} 0 0", f"{iss // 2} 0 0  {k} {k} 0", f"{k} {k} 0  0 0 0"]


def test_bulk_path_keeps_its_kz_segments(tmp_path, monkeypatch):
    """A 3D output is not cut (cubic PbTiO3, real output)."""
    shutil.copy(Path(__file__).parent / "data" / "opt_restart" / "tqb_pto.out", tmp_path)
    monkeypatch.setattr(d3gen.D3Generator, "_copy_wavefunction", lambda self: True)
    gen = d3gen.D3Generator(str(tmp_path / "tqb_pto.out"), "BAND", output_dir=str(tmp_path))
    assert gen.structure_info["dimensionality"] == 3
    gen.generate_d3(shared_config=dict(ROUTES["vectors"], auto_path=True, n_points=1000))
    deck = next(tmp_path.glob("*.d3")).read_text().splitlines()
    assert any(int(rec.split()[2]) or int(rec.split()[5]) for rec in deck[3:-1])


def test_in_plane_path_cuts_labels_with_segments():
    from d3_kpoints import in_plane_path
    segs = [[0, 0, 0, .5, 0, 0], [.5, 0, 0, 0, 0, .5], [0, 0, .5, 0, 0, 0], [0, 0, 0, 0, .5, 0]]
    labels = ["GAMMA", "X", "Z", "GAMMA", "Y"]
    kept, cut = in_plane_path(segs, labels)
    assert kept == [segs[0], segs[3]]
    assert cut == ["GAMMA", "X", "|", "GAMMA", "Y"]
    assert in_plane_path(segs, ["GAMMA", "X"]) == (kept, None)
