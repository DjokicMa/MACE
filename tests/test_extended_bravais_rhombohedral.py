"""Rhombohedral groups get SeeK-path's hR1/hR2 choice.

SeeK-path (HPKOT, Hinuma et al. 2017, Table 2; seekpath/hpkot) takes hR1 when
sqrt(3) a < sqrt(2) c in the hexagonal axes, i.e. c/a > sqrt(3/2) = 1.2247,
and hR2 otherwise; in rhombohedral axes that is alpha < 90 degrees. The
static lookup took hR1 for c/a < sqrt(6) = 2.449, so most real rhombohedral
crystals (c/a 2.5-7: AgBr R-3c 2.70, Bi2Se3, corundum) got hR2, and flat
cells with c/a < 1.22 got hR1.
"""
from pathlib import Path

import pytest

import d3_kpoints
from d3_kpoints import get_extended_bravais, get_seekpath_full_kpath

DATA = Path(__file__).parent / "data" / "opt_restart"
R_GROUPS = [146, 148, 155, 160, 161, 166, 167]
HEX_CELLS = [(5.1, 5.1, 4.0), (5.1, 5.1, 5.0), (5.1, 5.1, 6.0), (5.1, 5.1, 6.4),
             (5.1, 5.1, 7.4), (5.1, 5.1, 12.0), (4.14, 4.14, 28.6)]

TEMPLATE = """ LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - CONVENTIONAL CELL
        A           B           C        ALPHA        BETA       GAMMA
 {:11.5f} {:11.5f} {:11.5f} {:11.5f} {:11.5f} {:11.5f}

"""


@pytest.mark.parametrize("a,b,c", HEX_CELLS)
def test_hexagonal_axes(a, b, c):
    expected = "hR1" if c / a > 1.5 ** 0.5 else "hR2"
    assert get_extended_bravais(166, "R", a, b, c, 90, 90, 120) == expected


@pytest.mark.parametrize("alpha,expected", [(55.77, "hR1"), (80.0, "hR1"), (95.0, "hR2"),
                                            (105.0, "hR2")])
def test_rhombohedral_axes(alpha, expected):
    assert get_extended_bravais(166, "R", 7.3, 7.3, 7.3, alpha, alpha, alpha) == expected


def test_agbr_output_is_hr1(monkeypatch):
    """AgBr R-3c (real CRYSTAL23 output): conventional a = 6.84, c = 18.46.
    SeeK-path itself gives hR1 for this output."""
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    _, info = get_seekpath_full_kpath(167, "R", str(DATA / "tqc_agbr.out"))
    assert info["extended_bravais"] == "hR1"


@pytest.mark.parametrize("a,b,c", HEX_CELLS)
def test_static_route_reads_the_output(tmp_path, monkeypatch, a, b, c):
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    out = tmp_path / "r.out"
    out.write_text(TEMPLATE.format(a, b, c, 90, 90, 120))
    _, info = get_seekpath_full_kpath(166, "R", str(out))
    assert info["lookup_key"] == ("hR1" if c / a > 1.5 ** 0.5 else "hR2")


@pytest.mark.parametrize("sg", R_GROUPS)
@pytest.mark.parametrize("a,b,c", HEX_CELLS[:6])
def test_hr_variant_matches_seekpath(sg, a, b, c, seekpath_path):
    result = seekpath_path(sg, (a, b, c, 90, 90, 120))
    assert get_extended_bravais(sg, "R", a, b, c, 90, 90, 120) == result["bravais_lattice_extended"]
