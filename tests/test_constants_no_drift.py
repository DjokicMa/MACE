"""Pins the byte-identical constants consolidation (item 6a) as a no-op.

mace/constants.py is the single source of truth for HARTREE_TO_EV and
BOHR_TO_ANGSTROM. These asserts use exact == (not approx): the canonical sites
already held these full-precision values, so re-pointing them must not change a
single bit. If a future edit re-derives or truncates the shared constant, this
fails immediately.

NOT covered here (deliberately deferred — see REMEDIATION_PLAN 6a): the ~30
truncated 27.2114 / 27.211386 literals inside the validated property_extractor
parser and the untested d3/plotting scripts. Replacing those changes output and
is a separate, regression-pinned precision fix.
"""
from mace.constants import (
    HARTREE_TO_EV, EV_TO_HARTREE, BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR,
)


def test_canonical_values_exact():
    assert HARTREE_TO_EV == 27.211386245988
    assert BOHR_TO_ANGSTROM == 0.52917721067
    assert EV_TO_HARTREE == 1.0 / 27.211386245988
    assert ANGSTROM_TO_BOHR == 1.0 / 0.52917721067


def test_unit_converter_base_keys_single_sourced():
    from mace.database.utils.units import UnitConverter
    assert UnitConverter.ENERGY_CONVERSIONS["ev"] == HARTREE_TO_EV
    assert UnitConverter.LENGTH_CONVERSIONS["angstrom"] == BOHR_TO_ANGSTROM
    assert UnitConverter.LENGTH_CONVERSIONS["a"] == BOHR_TO_ANGSTROM


def test_dat_file_processor_constant_single_sourced():
    from mace.utils import dat_file_processor
    assert dat_file_processor.HARTREE_TO_EV == HARTREE_TO_EV


def test_units_derived_keys_unchanged():
    """The derived/scaled keys are intentionally left as literals (re-deriving
    via arithmetic risks last-ULP float drift). Pin that they are untouched."""
    from mace.database.utils.units import UnitConverter
    assert UnitConverter.ENERGY_CONVERSIONS["mev"] == 27211.386245988
    assert UnitConverter.LENGTH_CONVERSIONS["nm"] == 0.052917721067
