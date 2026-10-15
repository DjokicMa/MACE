"""num_imaginary_frequencies counts imaginary modes as the run analysis does.

The extractor counted every printed row with a negative frequency: the
rigid-body modes of a molecule (-0.0072 cm-1), numerical noise of a few
cm-1, and a degenerate row ("1-   3") once. It now uses the run analysis'
rule (mace/utils/imaginary_modes.imaginary, shared with the GUI): drop the
3 (periodic), 6 (molecule) or 5 (linear molecule) modes closest to zero,
then count the negative modes, degeneracy expanded, of at least 20 cm-1.
"""
from conftest import find_data


def _count(extractor, pattern):
    content = find_data(pattern).read_text(errors="ignore")
    return extractor._extract_frequency_properties(content)["num_imaginary_frequencies"]


MOL = "_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_freq_HSESOL3C_optimized.out"
BULK = "_opt_B3LYP-D3-D3_optimized_freq_B3LYP-D3-D3_optimized.out"


def test_molecule_rigid_body_and_noise_modes_not_counted(extractor):
    # -11.07 cm-1 (noise) and a -0.0072 cm-1 rigid-body mode: none
    assert _count(extractor, "FREQ/1LiFSI-1DEC-conf2" + MOL) == 0


def test_molecule_counts_the_real_imaginary_mode(extractor):
    # -23.89 and -13.69 cm-1: one beyond the noise threshold
    assert _count(extractor, "FREQ/1LiFSI-1EMS-conf3" + MOL[:-4] + "_temp.out") == 1


def test_degenerate_row_counts_each_mode(extractor):
    # one row "1-   3" at -164.23 cm-1: three modes
    assert _count(extractor, "FREQ/TiPbO3_mp-19845_sg221_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis."
                             "triplezeta" + BULK) == 3


def test_periodic_noise_not_counted(extractor):
    # a degenerate pair at -8.16 cm-1
    assert _count(extractor, "FREQ/Ag2Br2_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta" + BULK) == 0


def test_one_rule_for_the_gui_and_the_extractor():
    from mace.gui import runinfo
    from mace.utils import imaginary_modes
    assert runinfo.imaginary is imaginary_modes.imaginary
