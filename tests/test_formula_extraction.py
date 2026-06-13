"""Regression tests for d12 formula extraction (commit 43820ded).

The BASISSET section atoms must not bleed into the geometry atom count — the
fix resets the accumulator at the BASISSET terminator, so e.g. Sulfolane stays
C4H8SO2 (not C4H8SO3) and the 1EC electrolyte ends at O7 (not O8).
"""
from mace.utils.formula_extractor import extract_formula_from_d12

from conftest import find_data


def test_sulfolane_formula():
    d12 = find_data("**/Suflolane_MOLECULE_OPT_symm_HSESOL3C*opt_HSESOL3C_optimized.d12")
    assert extract_formula_from_d12(d12) == "C4H8SO2"


def test_ethylene_carbonate_formula():
    d12 = find_data("**/EC_MOLECULE_OPT_symm_HSESOL3C*opt_HSESOL3C_optimized.d12")
    assert extract_formula_from_d12(d12) == "C3H4O3"


def test_combined_electrolyte_oxygen_count():
    """The combined LiFSI+EC d12 must count O7 (BASISSET reset), not O8."""
    d12 = find_data("**/1LiFSI-1EC-conf1_MOLECULE_OPT_symm_HSESOL3C*opt_HSESOL3C_optimized.d12")
    formula = extract_formula_from_d12(d12)
    assert formula == "LiC3NH4S2F2O7"
    assert "O8" not in formula


def test_accepts_str_path_not_just_path():
    """Callers (e.g. the queue manager) pass a plain str; it must not raise
    AttributeError and must give the same result as a Path."""
    d12 = find_data("**/EC_MOLECULE_OPT_symm_HSESOL3C*opt_HSESOL3C_optimized.d12")
    from_path = extract_formula_from_d12(d12)
    from_str = extract_formula_from_d12(str(d12))   # would AttributeError before the fix
    assert from_str == from_path == "C3H4O3"


def test_missing_str_path_returns_none_not_raises():
    """A non-existent str path returns None (graceful), not an exception."""
    assert extract_formula_from_d12("/nonexistent/does_not_exist.d12") is None

