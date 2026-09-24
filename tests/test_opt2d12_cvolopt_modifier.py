"""A parent's CVOLOPT is kept with the optimization type it modifies.

CVOLOPT is not an optimization type of its own: the CRYSTAL23 manual (sec.
7.3) says it is "Only active with CELLONLY (cell parameters only
optimization), FULLOPTG (...) or INTREDUN". The parser took it as the type,
so a CELLONLY + CVOLOPT or INTREDUN + CVOLOPT parent got a child with
CVOLOPT alone, which CRYSTAL runs as a full constant-volume optimization.
"""
import io

import pytest

from d12_calc_basic import write_optimization_section
from d12_parsers import CrystalInputParser


def _parse(tmp_path, dft_lines, title="t", after=(), opt=(), basis=("BASISSET", "POB-TZVP")):
    deck = tmp_path / "p.d12"
    geometry = ["CRYSTAL", "0 0 0", "227", "3.567", "1", "6 0.125 0.125 0.125"]
    optgeom = ["OPTGEOM", *opt, "ENDOPT"] if opt else []
    deck.write_text("\n".join([title, *geometry, *optgeom, "END", *basis, "DFT",
                               *dft_lines, "ENDDFT", *after, "TOLDEE", "7",
                               "SHRINK", "8 8", "END"]) + "\n")
    return CrystalInputParser(str(deck)).parse()



@pytest.mark.parametrize("base", ["CELLONLY", "FULLOPTG", "INTREDUN"])
def test_cvolopt_modifies_the_parents_type(tmp_path, base):
    data = _parse(tmp_path, ["PBE0"], opt=[base, "CVOLOPT", "MAXCYCLE", "800"])
    opt = data["optimization_settings"]
    assert opt["type"] == base and opt["cvolopt_type"] == base
    buf = io.StringIO()
    write_optimization_section(buf, "FULLOPTG", opt, fill_missing_tolerances=False)
    assert buf.getvalue().split("\n")[:3] == ["OPTGEOM", base, "CVOLOPT"]


def test_cvolopt_is_not_carried_to_another_type(tmp_path):
    opt = _parse(tmp_path, ["PBE0"], opt=["CELLONLY", "CVOLOPT"])["optimization_settings"]
    buf = io.StringIO()
    write_optimization_section(buf, "FULLOPTG", {**opt, "type": "ATOMONLY"},
                               fill_missing_tolerances=False)
    assert "CVOLOPT" not in buf.getvalue().split("\n")


def test_cvolopt_alone_is_still_the_cvolopt_type(tmp_path):
    opt = _parse(tmp_path, ["PBE0"], opt=["CVOLOPT"])["optimization_settings"]
    assert opt["type"] == "CVOLOPT" and "cvolopt_type" not in opt


