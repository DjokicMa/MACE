"""Net Mulliken charges of pseudopotential (ECP) atoms use the valence charge.

The processor computed net charge = Z column - population. For an atom with a
pseudopotential, CRYSTAL's Mulliken Z column holds the conventional label
(282 for Pb with an input ECP, measured on test/SP/TiPbO3_*) and the
population counts valence electrons only, so Pb came out near +261. The
valence charge is the "NUCLEAR CHARGE" of CRYSTAL's PSEUDOPOTENTIAL
INFORMATION block ("ATOMIC NUMBER  82, NUCLEAR CHARGE 22.000"). The excerpt
below copies that layout; the numbers are made up.
"""
import pytest

PP_AND_MULLIKEN = """
 *******************************************************************************
 *** PSEUDOPOTENTIAL INFORMATION ***
 *******************************************************************************

 ATOMIC NUMBER  82, NUCLEAR CHARGE 22.000, GIVEN IN INPUT       PSEUDOPOTENTIAL
   TYPE       EXPONENT      COEFF.    N      EXPONENT      COEFF.    N
  P0 TMS    12.2963030  281.2854990   0     8.6326340   62.5202170   0

 MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
 MULLIKEN POPULATION ANALYSIS - NO. OF ELECTRONS   37.000000

  ATOM    Z CHARGE  A.O. POPULATION

   1 O    8  8.900  2.000  1.900  1.000  1.000  1.000  1.000  1.000
   2 PB 282 20.715  0.069 -0.626  2.582  1.833 -0.207 -0.209 -0.208  2.200
   3 O    8  7.385  2.000  1.385  1.000  1.000  1.000  0.500  0.500

 OVERLAP POPULATION CONDENSED TO ATOMS FOR FIRST NEIGHBORS (ELECTRONS)
"""


def test_valence_charges_read_the_pseudopotential_block():
    from mace.utils.population_analysis_processor import valence_charges
    assert valence_charges(PP_AND_MULLIKEN) == {82: 22.0}
    assert valence_charges("no pseudopotentials here") == {}


def test_ecp_atom_net_charge_uses_the_valence_charge(extractor, tmp_path):
    out = tmp_path / "pbo2.out"
    out.write_text(PP_AND_MULLIKEN)
    props = extractor.extract_all_properties(out)
    charges = {c["atom_number"]: c["mulliken_charge"] for c in props["processed_atomic_charges"]}
    assert charges[2] == pytest.approx(22.0 - 20.715)
    # all-electron atoms are unchanged
    assert charges[1] == pytest.approx(8 - 8.900)
    assert charges[3] == pytest.approx(8 - 7.385)


def test_processor_without_valence_data_keeps_plain_atoms():
    from mace.utils.population_analysis_processor import PopulationAnalysisProcessor
    res = PopulationAnalysisProcessor().process_population_data(
        {"mulliken_population": {"atoms": [
            {"atom_number": 1, "element": "C", "atomic_number": 6, "mulliken_charge": 6.1}]}})
    assert res["atomic_charges"][0]["mulliken_charge"] == pytest.approx(-0.1)


def test_two_hundred_label_without_valence_data_is_not_read_as_z_282():
    from mace.utils.population_analysis_processor import PopulationAnalysisProcessor
    res = PopulationAnalysisProcessor().process_population_data(
        {"mulliken_population": {"atoms": [
            {"atom_number": 1, "element": "PB", "atomic_number": 282, "mulliken_charge": 20.7}]},
         "mulliken_valence_charges": {"82": 22.0}})
    assert res["atomic_charges"][0]["mulliken_charge"] == pytest.approx(1.3)
