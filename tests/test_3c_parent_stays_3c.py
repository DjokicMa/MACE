"""A 3c composite method must come out of a child deck exactly as it went in.

HSEsol-3c, PBEh-3c and the other 3c methods include their own D3 and gCP terms,
so every 3c output prints a "DFT-D3-GCP DISPERSION AND COUNTERPOISE CORRECTION"
block. CrystalOutputParser read that block as an added D3 keyword and returned
"HSESOL3C-D3", and the SP, FREQ and OPT2 decks generated from an HSEsol-3c SP
parent then asked CRYSTAL for "HSESOL3C-D3".
"""
import pytest

from conftest import TEST_DATA
from d12_parsers import CrystalOutputParser
from test_numbered_calc_keeps_parent_method import _engine, _numbered, _only_deck

OUT_3C = """\
 WARNING **** RDDFTP **** HSEsol-3c has to be combined with sol-def2-mSVP basis set
 INFORMATION **** DFTD3 ENERGY **** CALCULATE DFTD3 ENERGY
 GCP ENERGY CALCULATION
 DFT-D3-GCP DISPERSION AND COUNTERPOISE CORRECTION
 D3 DISPERSION ENERGY (AU)      -2.4617798296766E-03
 GCP ENERGY (AU)                 3.8061220651320E-02
"""


@pytest.mark.parametrize("name,keyword", [("HSEsol-3c", "HSESOL3C"), ("PBEh-3c", "PBEH3C"),
                                          ("PBEsol0-3c", "PBESOL03C"), ("B97-3c", "B973C")])
def test_output_parser_does_not_add_d3_to_a_3c_method(tmp_path, name, keyword):
    lines = OUT_3C.replace("HSEsol-3c", name).split("\n")
    parser = CrystalOutputParser("unused.out")
    parser._extract_functional(lines)   # parse() order: functional, then dispersion
    parser._extract_dispersion(lines)
    data = parser.data
    assert data["functional"] == keyword
    assert not data.get("dispersion")


SP_3C = TEST_DATA / "SP" / ("FEC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C"
                            "_optimized_sp_HSESOL3C_optimized")


@pytest.mark.parametrize("target,base", [("SP2", "sp"), ("FREQ", "freq"), ("OPT2", "opt")])
def test_children_of_an_hsesol3c_sp_parent_keep_hsesol3c(tmp_path, target, base):
    """Real SP parent, real script: OPT -> SP -> {SP2, FREQ, OPT2}."""
    if not SP_3C.with_suffix(".d12").exists() or not SP_3C.with_suffix(".out").exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    eng = _engine(tmp_path, SP_3C.with_suffix(".d12").read_text(),
                  SP_3C.with_suffix(".out").read_text())
    eng.db.calc["calc_type"] = "SP"
    name, deck = _only_deck(_numbered(eng, target, True))
    body = deck.splitlines()
    assert body[body.index("DFT") + 1:body.index("ENDDFT")] == ["SPIN", "HSESOL3C", "XLGRID"], deck
    assert "-D3" not in deck
    assert name.endswith(f"_{base}_HSESOL3C_optimized.d12"), name
