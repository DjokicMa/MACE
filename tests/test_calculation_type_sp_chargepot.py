"""calculation_type is stored for SP and CHARGE+POTENTIAL runs too.

It was set to 'single_point' only when the output held 'SINGLE POINT
CALCULATION' or 'SCFDIR', which a CRYSTAL23 SP prints neither of, and never
for an ECH3/POT3 run, so both stored none. An SCF run that is no geometry
optimization (and no FREQ) is 'SP'; an ECH3/POT3 run 'CHARGE+POTENTIAL'.
Since 1.1.6 the values are the calculation types of the database's
calculations (OPT and SP were 'geometry_optimization' and 'single_point').
"""
import pytest

from conftest import find_data


@pytest.mark.parametrize("path,expected", [
    ("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out", "SP"),
    ("ECH3POT3/*.out", "CHARGE+POTENTIAL"),
    ("OPT/1_dia_opt_rev1.out", "OPT"),
    ("BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.out", "BAND"),
    ("DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out", "DOSS"),
    ("FREQ/1LiFSI-1DEC-conf1_*_freq_HSESOL3C_optimized.out", "FREQ"),
])
def test_calculation_type(extractor, path, expected):
    assert extractor.extract_all_properties(find_data(path)).get("calculation_type") == expected
