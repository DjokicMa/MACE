"""The k-path facts of a band plot's Inspector › Brillouin zone (1.1.6,
mace/gui/results.py kpath_info) and the zone's dimensionality from a properties
run's output (Crystal_d3/d3_bz.py: CRYSTAL's formal 500 Å makes a slab's zone
2D and a polymer's 1D). On the bundled example suite and tests/data/f25."""
import lzma
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mace.gui import results  # noqa: E402

SUITE = REPO / "examples" / "suite"


def put(src: Path, dst: Path):
    xz = src.with_name(src.name + ".xz")
    if xz.is_file():
        dst.write_bytes(lzma.decompress(xz.read_bytes()))
    elif src.is_file():
        dst.write_bytes(src.read_bytes())
    else:
        pytest.skip("the bundled example suite is not here")


def test_deck_band_dat_and_fort25(tmp_path):
    band = SUITE / "2-dense-3d" / "diamond" / "BAND"
    for name in ("diamond_band.BAND.DAT", "diamond_band.d3", "diamond_band.f25"):
        put(band / name, tmp_path / name)
    info = results.kpath_info(tmp_path / "diamond_band.BAND.DAT")
    assert info["source"] == "diamond_band.d3" and info["iss"] == 8 and info["points"] == 1000 and info["lines"] == 6
    assert info["segments"][1] == {"from": "X", "to": "U", "k_from": [0.5, 0.0, 0.5], "k_to": [0.625, 0.25, 0.625]}
    # without the deck: the fort.25's own records (one spin; the ends in units of an ISS it does not record)
    (tmp_path / "diamond_band.d3").unlink()
    f = results.kpath_info(tmp_path / "diamond_band.f25")
    assert f["source"] == "diamond_band.f25" and f["iss"] is None and f["lines"] == 6 and f["points"] == 1001
    assert f["segments"][0]["ints"] == [[0, 0, 0], [4, 0, 4]] and f["segments"][0]["k_from"] is None


def test_band_dat_header(tmp_path):
    put(SUITE / "4-polymer-1d" / "polyacetylene" / "BAND" / "polyacetylene_band.BAND.DAT", tmp_path / "p.BAND.DAT")
    info = results.kpath_info(tmp_path / "p.BAND.DAT")
    assert info["source"] == "p.BAND.DAT" and info["iss"] == 2 and info["points"] == 1000
    assert info["segments"] == [{"from": "GAMMA", "to": "", "k_from": [0.0, 0.0, 0.0], "k_to": [0.5, 0.0, 0.0]}]


def test_phonon_fort25_iss_from_its_deck(tmp_path):
    for name in ("dia_phonon_f25.fort.25", "dia_phonon_f25.d12"):
        shutil.copy2(REPO / "tests" / "data" / "f25" / name, tmp_path / name)
    info = results.kpath_info(tmp_path / "dia_phonon_f25.fort.25")
    assert info["domain"] == "phonon" and info["iss"] == 8 and info["lines"] == 4 and info["points"] == 197
    assert [s["k_to"] for s in info["segments"]] == [[0.5, 0.0, 0.5], [0.5, 0.25, 0.75], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]


@pytest.mark.parametrize("run, dim", [("4-polymer-1d/polyacetylene/BAND/polyacetylene_band", 1),
                                      ("3-slab-2d/graphene/BAND/graphene_band", 2),
                                      ("2-dense-3d/diamond/BAND/diamond_band", 3)])
def test_zone_dimensionality_of_a_properties_run(tmp_path, run, dim):
    pytest.importorskip("scipy")                                     # the zone is a Voronoi cell (scipy.spatial)
    src = SUITE / run
    for ext in ("BAND.DAT", "d3", "out"):
        put(src.with_name(f"{src.name}.{ext}"), tmp_path / f"b.{ext}")
    data = results.bz_data(tmp_path / "b.BAND.DAT")
    assert data["bz"]["dim"] == dim and data["path"], data["bz"]["dim"]
