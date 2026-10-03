"""An ECP atom whose valence charge the .out does not give is flagged, not guessed.

A Mulliken Z column of 200 + Z marks a valence-only (pseudopotential) atom:
the real outputs of the test corpus print Ag as 247 with POB-TZVP-REV2 and
with the explicit triple-zeta file, both with "ATOMIC NUMBER  47, NUCLEAR
CHARGE 19.000" in the PSEUDOPOTENTIAL INFORMATION block. When that charge is
missing, the old fallback measured the population against Z (47 for Ag), a
wrong charge printed as if it were a result (and, in opt2d12 without a
parent deck, a wrong DOPING). Now the charge is left unset and flagged.
"""
import pytest

from charge_records import charge_from_parent_out, out_electron_offset
from mace.utils.ion_fragments import fragment_charges, valence_of

# Ag+ and a Cl- far apart, in CRYSTAL23's Mulliken layout; the
# PSEUDOPOTENTIAL INFORMATION block (Ag: NUCLEAR CHARGE 19.000) is left out
OUT = """
 GEOMETRY FOR WAVE FUNCTION - DIMENSIONALITY OF THE SYSTEM    0
 *******************************************************************************
 ATOMS IN THE ASYMMETRIC UNIT    2 - ATOMS IN THE UNIT CELL:    2
     ATOM             X(ANGSTROM)         Y(ANGSTROM)         Z(ANGSTROM)
 *******************************************************************************
      1 T 247 AG     0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
      2 T  17 CL     8.000000000000E+00  0.000000000000E+00  0.000000000000E+00

 MULLIKEN POPULATION ANALYSIS - NO. OF ELECTRONS   36.000000

  ATOM    Z CHARGE  A.O. POPULATION

   1 AG 247 18.100  2.000  2.000
   2 CL  17 17.900  2.000  2.000

 OVERLAP POPULATION CONDENSED TO ATOMS FOR FIRST NEIGHBORS (ELECTRONS)
"""
PP = """
 *** PSEUDOPOTENTIAL INFORMATION ***
 ATOMIC NUMBER  47, NUCLEAR CHARGE 19.000, GIVEN IN INPUT       PSEUDOPOTENTIAL
"""


def test_valence_of_an_ecp_label_without_data_is_unknown():
    assert valence_of(247, {}) is None
    assert valence_of(247, {47: 19.0}) == 19.0
    assert valence_of(17, {}) == 17.0
    assert valence_of(147, {}) == 47.0  # 100 + Z: a second all-electron basis
    # 1000 + Z is all-electron too (manual p.25); 300 + Z a second ECP basis
    assert valence_of(1006, {}) == 6.0 and valence_of(1047, {}) == 47.0
    assert valence_of(347, {}) is None


def test_fragment_charges_flag_the_unknown_valence():
    res = fragment_charges(OUT, force=True)
    count = res["electron_count"]
    assert count["net_charge"] is None and count["valence_sum"] is None
    assert count["valence_unknown"] == ["Ag (atom 1)"]
    ag = next(e for e in res["fragment_charges"] if e["formula"] == "Ag")
    assert ag["mulliken_charge"] is None and ag["deviation"] is None
    assert "valence charge" in ag["flag"]
    cl = next(e for e in res["fragment_charges"] if e["formula"] == "Cl")
    assert cl["mulliken_charge"] == pytest.approx(-0.9)


def test_with_the_valence_data_the_charge_is_computed():
    res = fragment_charges(PP + OUT, force=True)
    ag = next(e for e in res["fragment_charges"] if e["formula"] == "Ag")
    assert ag["mulliken_charge"] == pytest.approx(0.9)


def test_the_report_prints_a_flag_not_a_number():
    from mace.utils.ion_fragments import format_fragment_charges
    text = "\n".join(format_fragment_charges(fragment_charges(OUT, force=True)))
    assert "Mulliken n/a" in text and "valence charge" in text
    assert "+28.9" not in text  # what the Z fallback printed (47 - 18.1)


def test_processor_flags_the_atom():
    from mace.utils.population_analysis_processor import PopulationAnalysisProcessor
    res = PopulationAnalysisProcessor().process_population_data(
        {"mulliken_population": {"atoms": [
            {"atom_number": 1, "element": "AG", "atomic_number": 247, "mulliken_charge": 18.1},
            {"atom_number": 2, "element": "Cl", "atomic_number": 17, "mulliken_charge": 17.9}]}})
    ag, cl = res["atomic_charges"]
    assert ag["mulliken_charge"] is None and ag["formal_charge"] is None
    assert "valence charge" in ag["flag"]
    assert cl["mulliken_charge"] == pytest.approx(-0.9) and "flag" not in cl


def test_opt2d12_from_the_out_alone_writes_no_guessed_doping():
    assert out_electron_offset(OUT) is None
    child = ("c\nMOLECULE\n1\n2\n47 0.0 0.0 0.0\n17 8.0 0.0 0.0\nBASISSET\nPOB-TZVP-REV2\n"
             "DFT\nPBE0\nENDDFT\nTOLDEE\n9\nEND\n")
    out, notes = charge_from_parent_out(child, OUT)
    assert "DOPING" not in out
    assert any(n.startswith("WARNING") and "valence charge" in n for n in notes), notes
