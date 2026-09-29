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


def test_standalone_fallback_values_match_canonical():
    """Scripts that also run without the mace package keep a literal fallback
    for the constants they import from mace.constants. Each fallback must be
    the canonical value exactly, so a standalone run computes the same numbers."""
    import ast
    import mace.constants as canonical
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    files = ["Crystal_d3/CRYSTALOptToD3.py", "Crystal_d3/d3_interactive.py",
             "mace/utils/advanced_electronic_analyzer.py"]
    for rel in files:
        tree = ast.parse((repo / rel).read_text())
        guards = [n for n in tree.body if isinstance(n, ast.Try)
                  and isinstance(n.body[0], ast.ImportFrom)
                  and n.body[0].module == "mace.constants"]
        assert len(guards) == 1, rel
        imported = [a.name for a in guards[0].body[0].names]
        ns = {}
        exec(compile(ast.Module(body=guards[0].handlers[0].body, type_ignores=[]),
                     rel, "exec"), ns)
        assert sorted(k for k in ns if not k.startswith("__")) == sorted(imported), rel
        for name in imported:
            assert ns[name] == getattr(canonical, name), (rel, name)
