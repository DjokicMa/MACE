"""magnetic_classification follows the run's spin, not the Mulliken print.

It was 'spin_polarized' whenever both Mulliken spin tables were printed and
'non_magnetic' otherwise, so a closed-shell molecule run unrestricted
(SUMMED SPIN DENSITY 0) was 'spin_polarized'. It is now 'spin_polarized'
for an unrestricted run with a net spin (alpha - beta electrons) or atomic
spin populations, and 'non_magnetic' for a restricted run or an
unrestricted one whose spin is zero. A run that prints no spin at all (a
FREQ restarted from a wavefunction) keeps the earlier rule.
"""
import pytest

from conftest import find_data


@pytest.mark.parametrize("path,expected", [
    # unrestricted, alpha - beta = 0
    ("SP/1LiFSI-1DEC-conf1_*_sp_HSESOL3C_optimized.out", "non_magnetic"),
    # the FSI radical: alpha - beta = -1
    ("SP/FSI_MOLECULE_relaxed_optimized_sp.out", "spin_polarized"),
    # a doublet FSI on graphene (alpha - beta = -1)
    ("SP/4LG_FSI_2x2_AA_FSI_opt_sp.out", "spin_polarized"),
    # unrestricted diamond FREQ without Mulliken tables (alpha = beta = 48)
    ("FREQ/1_dia_opt_rev1_freq_B3LYP-D3-D3_optimized_supercel222.out", "non_magnetic"),
    # a FREQ restarted from a wavefunction prints no spin (no SCF): as before
    ("FREQ/1LiFSI-1DEC-conf1_*_freq_HSESOL3C_optimized.out", "non_magnetic"),
    # restricted closed shell
    ("OPT/1_dia_opt_BULK_OPTGEOM.out", "non_magnetic"),
])
def test_magnetic_classification(extractor, path, expected):
    props = extractor.extract_all_properties(find_data(path))
    assert props["magnetic_classification"] == expected
