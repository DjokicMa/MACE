"""The deck title (line 1) is free text, never a keyword.

A title that happened to be a keyword was read as one by the remaining
input-deck scans: "SMEAR" gave the child Fermi smearing the parent never
had, "BASISSET" made the basis set "CRYSTAL" (the line after the title),
and "FREQCALC", "SPINLOCK", "TOLINTEG", "SHRINK", "MAXCYCLE" and the like
were read with the next line as their value.
"""
import pytest

from d12_parsers import CrystalInputParser


def _parse(tmp_path, dft_lines, title="t", after=(), opt=(), basis=("BASISSET", "POB-TZVP")):
    deck = tmp_path / "p.d12"
    geometry = ["CRYSTAL", "0 0 0", "227", "3.567", "1", "6 0.125 0.125 0.125"]
    optgeom = ["OPTGEOM", *opt, "ENDOPT"] if opt else []
    deck.write_text("\n".join([title, *geometry, *optgeom, "END", *basis, "DFT",
                               *dft_lines, "ENDDFT", *after, "TOLDEE", "7",
                               "SHRINK", "8 8", "END"]) + "\n")
    return CrystalInputParser(str(deck)).parse()



def test_title_smear_adds_no_smearing(tmp_path):
    data = _parse(tmp_path, ["PBE0"], title="SMEAR")
    assert not data.get("use_smearing")


def test_title_basisset_is_not_the_basis_keyword(tmp_path):
    data = _parse(tmp_path, ["PBE0"], title="BASISSET")
    assert data["basis_set"] == "POB-TZVP"


@pytest.mark.parametrize("title, key", [
    ("FREQCALC", "freq_settings"), ("SPINLOCK", "spinlock"),
    ("TOLINTEG", "tolerances"), ("SCFDIR", "scf_direct"), ("HISTDIIS", "diis_history"),
])
def test_other_keyword_titles_are_not_read(tmp_path, title, key):
    data = _parse(tmp_path, ["PBE0"], title=title)
    assert not data.get(key) or key == "tolerances" and "TOLINTEG" not in data[key]
    assert data.get("calculation_type") != "FREQ"
