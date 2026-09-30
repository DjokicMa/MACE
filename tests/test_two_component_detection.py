"""Telling a two-component (TWOCOMPON) run from a one-component one.

A 2c-SCF is a single point to CRYSTAL23 (no 2c optimisation or frequency run,
manual p. 166), so it is a flag, not a calculation type. Its .out is
recognised by lines CRYSTAL prints for a 2c-SCF - the per-cycle X and Y
magnetization components and the occupied-spinor count. The lines below are
copied from the .out of an fcc Au 2c-SCF with SOC (stock CRYSTAL23 1.0.1,
HPCC); the 1c outputs are the real graphene/polyyne runs in tests/data.
"""
import sqlite3
from pathlib import Path

import pytest

from mace.utils import calc_detection as C

DATA = Path(__file__).resolve().parent / "data" / "low_dim_groups"

# From Au_soc.out (HPCC), cycle 0 of the 2c-SCF.
AU_2C_LINES = """ CYC   0 ETOT(AU) -1.355487894140E+02 DETOT -1.36E+02 tst  0.00E+00 PX  1.00E+00
 TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT FDIK        TELAPSE       12.53 TCPU       11.77
 POSSIBLY CONDUCTING STATE - EFERMI(AU) -7.0721845E-02 (RES. CHARGE  3.51E-11;IT. 10)
 - NUMBER OF FULLY OCCUPIED/TOTAL SPINORS -    12 /     70
 TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT PDIG        TELAPSE       12.77 TCPU       12.01
 CHARGE NORMALIZATION FACTOR   1.00000000
 ATOMIC PARTICLE-No DENSITY:
  19.0000000
 TOTAL X-COMP MAGNETIZATION    0.00000001
 ATOMIC MX                 :
   0.0000000
 TOTAL Y-COMP MAGNETIZATION   -0.00000001
 ATOMIC MY                 :
  -0.0000000
 TOTAL Z-COMP MAGNETIZATION    0.00000000
 ATOMIC MZ                 :
   0.0000000
"""
ONE_COMPONENT = sorted(DATA.glob("*.out"))


def test_a_2c_output_is_recognised_by_its_own_lines():
    assert C.is_two_component_output(AU_2C_LINES)
    # each kind of line on its own
    spinors = " - NUMBER OF FULLY OCCUPIED/TOTAL SPINORS -    12 /     70\n"
    magnet = " TOTAL X-COMP MAGNETIZATION    0.00000001\n"
    assert C.is_two_component_output(spinors) and C.is_two_component_output(magnet)


@pytest.mark.parametrize("out", ONE_COMPONENT, ids=lambda p: p.name)
def test_real_1c_outputs_are_not_2c(out):
    assert ONE_COMPONENT, "tests/data/low_dim_groups has no .out"
    assert not C.is_two_component_output(out.read_text())


def test_the_title_is_not_read():
    # MACE titles decks after the file name, and CRYSTAL echoes the title
    title = " Au_TWOCOMPON_TOTAL_X-COMP_MAGNETIZATION_SPINORS_sp\n"
    assert not C.is_two_component_output(title)


SOC_DECK = """Au_BULK_sp
CRYSTAL
0 0 0
225
4.0782
1
279 0.0 0.0 0.0
END
99 0
END
DFT
PBE0
ENDDFT
SHRINK
12 12
TWOCOMPON
SOC
END
SCFDIR
END
"""
# A 2c deck without SOC, closed by ENDTWO as in the manual's p. 172 examples.
TWOC_NO_SOC = SOC_DECK.replace("TWOCOMPON\nSOC\nEND\n",
                               "TWOCOMPON\nGUESSPATNC\n1\n1 90.0 90.0 1.0\nENDTWO\n")
SCALAR_DECK = SOC_DECK.replace("TWOCOMPON\nSOC\nEND\n", "")
TITLE_ONLY = "Au_TWOCOMPON\n" + SCALAR_DECK.split("\n", 1)[1]


@pytest.mark.parametrize("deck,two_c,soc", [
    (SOC_DECK, True, True), (TWOC_NO_SOC, True, False),
    (SCALAR_DECK, False, False), (TITLE_ONLY, False, False),
], ids=["soc", "2c-no-soc", "scalar", "title-only"])
def test_deck_flags(deck, two_c, soc):
    assert C.is_two_component_deck(deck) is two_c
    assert C.is_spin_orbit_deck(deck) is soc
    # still a single point
    assert C.d12_calc_type(C.deck_records(deck)) == "SP"


# --- recorded in the database like the other SCF flags -------------------------

def _extractor(tmp_path):
    pytest.importorskip("mace.database.materials")
    from mace.utils.property_extractor import CrystalPropertyExtractor
    return CrystalPropertyExtractor(db_path=str(tmp_path / "materials.db"))


def test_the_scf_settings_carry_the_2c_flag_only_for_a_2c_run(tmp_path):
    ext = _extractor(tmp_path)
    assert ext._extract_scf_settings(AU_2C_LINES)["scf_two_component"] is True
    for out in ONE_COMPONENT:
        assert "scf_two_component" not in ext._extract_scf_settings(out.read_text())


def test_a_2c_run_is_stored_with_the_flag(tmp_path):
    ext = _extractor(tmp_path)
    out = tmp_path / "Au_soc.out"
    out.write_text(AU_2C_LINES + " == SCF ENDED - CONVERGENCE ON ENERGY      "
                   "E(AU) -1.3586079676234E+02 CYCLES  55\n")
    props = ext.extract_all_properties(out, material_id="Au_soc")
    assert props["scf_two_component"] is True
    ext.save_properties_to_database(props)
    with sqlite3.connect(tmp_path / "materials.db") as conn:
        rows = conn.execute("SELECT property_value, property_value_text FROM properties "
                            "WHERE material_id = 'Au_soc' AND property_name = "
                            "'scf_two_component'").fetchall()
    assert rows == [(1.0, "True")]
