"""Corpus-free tests for the calculation-type helpers in mace/utils/calc_detection.py.

The sweeps that check every deck and output under ``test/`` skip in CI. These
pin the helpers directly with inline decks written in CRYSTAL23 syntax and
output lines in the form CRYSTAL23 prints them: which record wins when a deck
holds several, which file-name token names the step, and which printed lines
mark an OPT, FREQ, BAND, DOSS, CHARGE+POTENTIAL or TRANSPORT run.
"""
from pathlib import Path

import pytest

from mace.utils.calc_detection import (
    CHARGE_POTENTIAL_RECORDS,
    PROPERTY_CALC_TYPES,
    calc_type_from_filename,
    d3_calc_type,
    d12_calc_type,
    deck_records,
    deck_records_for,
    is_band_output,
    is_charge_potential_output,
    is_doss_output,
    is_frequency_output,
    is_optimization_output,
    is_transport_output,
)

# --- Inline decks ------------------------------------------------------------

GEOMETRY = "CRYSTAL\n0 0 0\n227\n3.5668\n1\n6 0.125 0.125 0.125\n"
BASIS_DFT_SCF = ("BASISSET\nPOB-TZVP-REV2\nDFT\nB3LYP-D3\nXLGRID\nENDDFT\n"
                 "TOLINTEG\n7 7 7 7 14\nTOLDEE\n7\nSHRINK\n8 16\nSCFDIR\nPPAN\nEND\n")

SP_D12 = "diamond_sp\n" + GEOMETRY + "END\n" + BASIS_DFT_SCF
OPT_D12 = ("diamond_opt\n" + GEOMETRY
           + "OPTGEOM\nFULLOPTG\nTOLDEG\n0.0003\nTOLDEX\n0.0012\nMAXCYCLE\n800\nENDOPT\nEND\n"
           + BASIS_DFT_SCF)
FREQ_D12 = ("diamond_freq\n" + GEOMETRY
            + "FREQCALC\nNUMDERIV\n2\nINTENS\nINTPOL\nENDFREQ\nEND\n" + BASIS_DFT_SCF)


# --- deck_records -----------------------------------------------------------

def test_d12_records_are_upper_cased_and_stripped():
    records = deck_records("title\n  optgeom \ncrystal\n")
    assert records == {"OPTGEOM", "CRYSTAL"}


def test_d12_title_is_dropped_even_when_it_is_a_keyword():
    """Line 1 of a .d12 is free text whatever it says."""
    assert deck_records("OPTGEOM\nCRYSTAL\nEND\n") == {"CRYSTAL", "END"}


def test_blank_lines_become_an_empty_record():
    assert "" in deck_records("t\nCRYSTAL\n\nEND\n")


def test_d3_band_title_is_skipped_only_after_a_band_record():
    deck = "NEWK\n12 12 12\n1 0\nDOSS\n0 1000 4 11 1 14 0\nBAND\nFREQCALC_title\n14 16 1000 1 259 1 0\nEND\n"
    records = deck_records(deck, is_d3=True)
    assert {"NEWK", "DOSS", "BAND", "END", "14 16 1000 1 259 1 0"} <= records
    assert "FREQCALC_TITLE" not in records


def test_d3_band_title_skip_matches_the_whole_record_only():
    """'BANDLIST' is not BAND, so the next line is not a title."""
    records = deck_records("BANDLIST\nDOSS\nEND\n", is_d3=True)
    assert "DOSS" in records


@pytest.mark.parametrize("path,is_d3", [
    ("x.d3", True), ("X.D3", True), (Path("dir.d12/x.d3"), True),
    ("x.d12", False), ("x.D12", False), ("x", False), ("x.d3.bak", False),
])
def test_deck_records_for_picks_the_kind_from_the_suffix(path, is_d3):
    content = "NEWK\nDOSS\n"
    assert deck_records_for(path, content) == deck_records(content, is_d3=is_d3)


# --- d12_calc_type ----------------------------------------------------------

@pytest.mark.parametrize("deck,expected", [
    (SP_D12, "SP"),
    (OPT_D12, "OPT"),
    (FREQ_D12, "FREQ"),
])
def test_d12_calc_type_of_real_style_decks(deck, expected):
    assert d12_calc_type(deck_records(deck)) == expected


def test_d12_calc_type_optgeom_wins_over_freqcalc():
    assert d12_calc_type({"OPTGEOM", "FREQCALC"}) == "OPT"


def test_d12_calc_type_ignores_keywords_in_the_title():
    deck = "x_OPTGEOM_FREQCALC\n" + GEOMETRY + "END\n" + BASIS_DFT_SCF
    assert d12_calc_type(deck_records(deck)) == "SP"


def test_d12_calc_type_needs_the_whole_record():
    assert d12_calc_type({"OPTGEOM2", "FREQCALCX", "ENDOPT"}) == "SP"
    assert d12_calc_type(set()) == "SP"


# --- d3_calc_type -----------------------------------------------------------

def test_property_calc_types_table():
    assert PROPERTY_CALC_TYPES == ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL",
                                   "MATDUMP")
    assert CHARGE_POTENTIAL_RECORDS == frozenset({"ECH3", "ECHG", "POT3", "POTC"})


@pytest.mark.parametrize("records,expected", [
    ({"BOLTZTRA", "ECH3", "DOSS", "BAND"}, "TRANSPORT"),
    ({"POT3", "DOSS", "BAND"}, "CHARGE+POTENTIAL"),
    ({"ECHG", "BAND"}, "CHARGE+POTENTIAL"),
    ({"DOSS", "BAND"}, "DOSS"),
    ({"BAND", "NEWK"}, "BAND"),
    ({"NEWK", "END"}, None),
    (set(), None),
])
def test_d3_calc_type_priority(records, expected):
    """TRANSPORT > CHARGE+POTENTIAL > DOSS > BAND, None when none is present."""
    assert d3_calc_type(records) == expected


def test_every_d3_calc_type_is_a_property_calc_type():
    for records in ({"BOLTZTRA"}, {"ECH3"}, {"DOSS"}, {"BAND"}):
        assert d3_calc_type(records) in PROPERTY_CALC_TYPES


def test_d3_calc_type_from_a_combined_newk_doss_band_deck():
    deck = ("NEWK\n12 12 12\n1 0\nBAND\nDOSS_in_title\n14 16 1000 1 259 1 0\n"
            "0 0 0 8 8 8\nEND\n")
    assert d3_calc_type(deck_records(deck, is_d3=True)) == "BAND"


# --- calc_type_from_filename ------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("mat_chargepot.d3", "CHARGE+POTENTIAL"),
    ("mat_charge_potential.d3", "CHARGE+POTENTIAL"),
    ("mat_potential.out", "CHARGE+POTENTIAL"),
    ("mat_cp.d3", "CHARGE+POTENTIAL"),
    ("mat_transp.out", "TRANSPORT"),
    ("mat_dos.d3", "DOSS"),
    ("mat_freq2.out", "FREQ"),
    ("mat_sp10.d12", "SP"),
    ("MAT_OPT.D12", "OPT"),                  # case-insensitive
    ("mat_opt.f9", "OPT"),
    ("mat_sp.f25", "SP"),
    ("mat_band.sh", "BAND"),
    ("mat_freq.log", "FREQ"),
    ("/scratch/run_sp/mat_opt.out", "OPT"),  # only the file name counts
    ("mat_opt_band_doss", "DOSS"),           # no suffix: the last token wins
    ("mat_opt.xyz", None),                   # unknown suffix is kept: 'opt.xyz'
    ("mat_spin.d12", None),                  # '_spin' is not '_sp'
    ("mat_bands.d3", None),
    ("mat_optb.d12", None),
    ("mat.d12", None),
])
def test_calc_type_from_filename_tokens(name, expected):
    assert calc_type_from_filename(name) == expected


def test_calc_type_from_filename_accepts_a_path():
    assert calc_type_from_filename(Path("a_sp") / "b_freq.out") == "FREQ"


# --- Output-file markers ------------------------------------------------------

# Lines in the form CRYSTAL23 prints them.
TITLE_ECHO = " ./mat_OPTGEOM_FREQCALC_BAND_DOSS_ECH3_BOLTZTRA_sp\n"

OPT_LINES = [
    " INFORMATION **** ATOMONLY **** ONLY ATOMIC POSITIONS OPTIMIZED\n",
    " INFORMATION **** CELLONLY **** ONLY CELL PARAMETERS OPTIMIZED\n",
    " INFORMATION **** ITATOCEL **** ITERATIVE ATOMIC AND CELL OPTIMIZATION\n",
    " INFORMATION **** CVOLOPT **** CONSTANT VOLUME OPTIMIZATION\n",
    " ATOMS OPTIMIZATION - POINT    2\n",
    " * OPT END - FAILED * E(AU):  -7.621718784637E+01  POINTS  800 *\n",
]
FREQ_LINES = [
    " INFORMATION **** READM2 **** FREQCALC - MULLIKEN POPULATION ANALYSIS NOT PERFORMED\n",
    " VIBRATIONAL FREQUENCIES (CM**-1) AND ASSIGNMENTS\n",
    "    MODES         EIGV          FREQUENCIES     IRREP  IR   INTENS    RAMAN\n",
]
TRANSPORT_LINES = [
    " SEEBECK COEFFICIENT DATA WRITTEN ON FILE SEEBECK.DAT\n",
]
DOSS_LINES = [
    " TOTAL AND PROJECTED DENSITY OF STATES - FOURIER LEGENDRE METHOD\n",
    " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT DOSS        TELAPSE        1.46 TCPU        0.80\n",
]
BAND_LINES = [
    " *  BAND STRUCTURE                                                             *\n",
    " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT BAND        TELAPSE        1.68 TCPU        0.92\n",
]
CP_LINES = [
    " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT ECH3 START  TELAPSE        1.06 TCPU        0.51\n",
    " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT ECHG        TELAPSE        2.00 TCPU        1.00\n",
    " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT POT3        TELAPSE      566.08 TCPU      513.93\n",
    " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT POTC        TELAPSE        3.00 TCPU        2.00\n",
]

DETECTORS = {
    "OPT": is_optimization_output,
    "FREQ": is_frequency_output,
    "TRANSPORT": is_transport_output,
    "DOSS": is_doss_output,
    "BAND": is_band_output,
    "CHARGE+POTENTIAL": is_charge_potential_output,
}
MARKERS = {
    "OPT": OPT_LINES,
    "FREQ": FREQ_LINES,
    "TRANSPORT": TRANSPORT_LINES,
    "DOSS": DOSS_LINES,
    "BAND": BAND_LINES,
    "CHARGE+POTENTIAL": CP_LINES,
}


@pytest.mark.parametrize("kind,line", [
    (kind, line) for kind, lines in MARKERS.items() for line in lines])
def test_each_marker_is_detected_by_its_own_detector_only(kind, line):
    text = TITLE_ECHO + line
    for other, detect in DETECTORS.items():
        assert detect(text) is (other == kind), (other, line)


@pytest.mark.parametrize("detect", list(DETECTORS.values()))
def test_title_echo_alone_is_no_marker(detect):
    assert detect(TITLE_ECHO) is False
    assert detect("") is False


def test_doss_band_range_line_is_not_a_band_marker():
    """A DOSS run prints 'FROM BAND n TO BAND m'; that is not BAND output."""
    line = " FROM BAND    4 TO BAND   11 ENERGY RANGE -0.36749E+00  0.73499E+00\n"
    assert not is_band_output(line)


def test_timing_line_needs_at_least_ten_t():
    assert not is_doss_output(" TTTTTTTTT DOSS        TELAPSE        1.46\n")
    assert is_doss_output(" TTTTTTTTTT DOSS        TELAPSE        1.46\n")


def test_optimization_marker_must_start_the_line():
    """Mid-line text (as in a title echo) never counts."""
    assert not is_optimization_output(" ./x FINAL OPTIMIZED GEOMETRY\n")
    assert is_optimization_output("FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM 3\n")


def test_frequency_banner_must_be_the_whole_line():
    assert not is_frequency_output(" FREQUENCY CALCULATION ENDED\n")
    assert is_frequency_output("   FREQUENCY CALCULATION   \n")
