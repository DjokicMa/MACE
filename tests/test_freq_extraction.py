"""Regression tests: the FREQ path must STORE the vibrational data it parses.

CrystalPropertyExtractor used to parse the MODES/FREQUENCIES table into a local
list that never reached ``props`` (the units-line anchor required the literal
``(CM**(-1))`` while real CRYSTAL output prints ``(CM**-1)``, and the
space-separated mode range ``1-   1`` raised ValueError before any row was
stored). As a result ``vibrational_frequencies`` came back ``None`` even though
thermodynamics (Gibbs/ZPE) populated fine.

These tests run the REAL extractor against real ``test/FREQ/*.out`` corpus
files and lock the contract that vibrational data is now extracted, while
thermodynamics keep working.
"""
import math
from pathlib import Path

from mace.utils.property_extractor import CrystalPropertyExtractor

from conftest import find_data


def _extract(pattern: str):
    out = find_data(pattern)
    ex = CrystalPropertyExtractor(enable_tracking=False)
    props = ex.extract_all_properties(Path(out))
    return props


def test_freq_stores_vibrational_frequencies():
    """Real molecular FREQ output: vibrational data is stored, not discarded."""
    props = _extract(
        "FREQ/EC_MOLECULE_OPT_symm_HSESOL3C*freq_HSESOL3C_optimized.out"
    )

    # Thermodynamics must still populate (preserve existing behavior).
    assert props.get("gibbs_free_energy_au") is not None
    assert props.get("has_frequency_data") is True

    # The vibrational list must now be present and non-empty.
    freqs = props.get("vibrational_frequencies")
    assert isinstance(freqs, list)
    assert len(freqs) > 0

    # Every parsed frequency is a finite float.
    for mode in freqs:
        f_cm = mode["frequency_cm"]
        assert isinstance(f_cm, float)
        assert math.isfinite(f_cm)

    # Aggregate count matches the list length and is positive.
    assert props.get("num_vibrational_modes") == len(freqs)
    assert props["num_vibrational_modes"] > 0

    # Frequencies live in a plausible cm**-1 range for a small molecule.
    real_freqs = [m["frequency_cm"] for m in freqs if m["frequency_cm"] > 0.1]
    assert real_freqs, "expected at least one genuine (non-translational) mode"
    assert max(real_freqs) < 5000.0  # C-H stretches top out around 3000-3500
    assert props.get("highest_frequency_cm") == max(real_freqs)
    assert props.get("lowest_frequency_cm") == min(real_freqs)

    # IR intensities are captured per mode (KM/MOL, in brackets in the table).
    assert any(m["ir_intensity"] > 0.0 for m in freqs)

    # num_imaginary_frequencies is always set when modes were parsed; this
    # well-converged optimum has none.
    assert props.get("num_imaginary_frequencies") == 0


def test_freq_detects_imaginary_mode():
    """A FREQ output with a negative frequency reports an imaginary mode."""
    props = _extract(
        "FREQ/1LiFSI-1FEC-conf2_MOLECULE_OPT_symm_HSESOL3C*freq_HSESOL3C_optimized.out"
    )
    freqs = props.get("vibrational_frequencies")
    assert isinstance(freqs, list) and len(freqs) > 0

    # Imaginary modes are negative cm**-1 (no trailing 'i').
    assert props.get("num_imaginary_frequencies", 0) >= 1
    assert any(m["frequency_cm"] < 0 for m in freqs)
    assert props["num_imaginary_frequencies"] == sum(
        1 for m in freqs if m["frequency_cm"] < 0
    )


def test_freq_periodic_grouped_degenerate_modes():
    """Periodic FREQ output groups degenerate modes into ranges (e.g. 1-3)."""
    props = _extract(
        "FREQ/Ag1Cl1_sym_CRYSTAL_OPT_symm*freq_B3LYP-D3-D3_optimized.out"
    )
    freqs = props.get("vibrational_frequencies")
    assert isinstance(freqs, list) and len(freqs) > 0

    # At least one row spans a degeneracy range (mode_end > mode_start).
    assert any(m["mode_end"] > m["mode_start"] for m in freqs)

    # Frequencies are finite floats and the count agrees with the list length.
    for m in freqs:
        assert math.isfinite(m["frequency_cm"])
    assert props.get("num_vibrational_modes") == len(freqs)
