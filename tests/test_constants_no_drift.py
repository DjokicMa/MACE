"""Pins the shared physical constants (mace/constants.py): CODATA 2018.

mace/constants.py is the single source of truth for HARTREE_TO_EV,
BOHR_TO_ANGSTROM and the factors derived from them and the exact SI
constants. These asserts use exact == (not approx): if a future edit
re-derives or truncates a shared constant, this fails immediately. 1.1.6
corrected BOHR_TO_ANGSTROM from the CODATA 2014 0.52917721067 to the CODATA
2018 0.529177210903, and every MACE module that hard-coded a Bohr, Hartree,
kJ/mol or cm⁻¹ factor now imports it from here (the guard below).

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
    assert BOHR_TO_ANGSTROM == 0.529177210903
    assert EV_TO_HARTREE == 1.0 / 27.211386245988
    assert ANGSTROM_TO_BOHR == 1.0 / 0.529177210903


def test_unit_converter_base_keys_single_sourced():
    from mace.database.utils.units import UnitConverter
    assert UnitConverter.ENERGY_CONVERSIONS["ev"] == HARTREE_TO_EV
    assert UnitConverter.LENGTH_CONVERSIONS["angstrom"] == BOHR_TO_ANGSTROM
    assert UnitConverter.LENGTH_CONVERSIONS["a"] == BOHR_TO_ANGSTROM


def test_dat_file_processor_constant_single_sourced():
    from mace.utils import dat_file_processor
    assert dat_file_processor.HARTREE_TO_EV == HARTREE_TO_EV


def test_units_derived_keys_unchanged():
    """The derived/scaled keys come from mace/constants.py too (they were
    CODATA 2014 literals: kJ/mol 2625.499638, cm-1 219474.631363, Bohr
    0.52917721067)."""
    import pytest
    from mace.database.utils.units import UnitConverter as UC
    assert UC.ENERGY_CONVERSIONS["mev"] == pytest.approx(27211.386245988, rel=1e-15)
    assert UC.LENGTH_CONVERSIONS["nm"] == pytest.approx(0.0529177210903, rel=1e-15)
    # CODATA 2018 values
    assert UC.ENERGY_CONVERSIONS["kj/mol"] == pytest.approx(2625.4996394799, rel=1e-13)
    assert UC.ENERGY_CONVERSIONS["kcal/mol"] == pytest.approx(627.5094740631, rel=1e-13)
    assert UC.ENERGY_CONVERSIONS["cm-1"] == pytest.approx(219474.6313632, rel=1e-13)
    assert UC.ENERGY_CONVERSIONS["thz"] == pytest.approx(6579.683920502, rel=1e-12)
    assert UC.PRESSURE_CONVERSIONS["gpa"] == pytest.approx(29421.015697, rel=1e-10)
    assert UC.FREQUENCY_CONVERSIONS["cm-1"] == pytest.approx(33.35640951981520, rel=1e-14)


def test_derived_constants_are_codata_2018():
    import pytest
    import mace.constants as C
    for name, value in (("HARTREE_TO_J", 4.3597447222071e-18), ("EV_TO_KJ_MOL", 96.48533212331002),
                        ("HARTREE_TO_KJ_MOL", 2625.4996394799), ("HARTREE_TO_KCAL_MOL", 627.5094740631),
                        ("EV_TO_CM1", 8065.543937349212), ("HARTREE_TO_CM1", 219474.6313632),
                        ("HARTREE_TO_THZ", 6579.683920502), ("THZ_TO_CM1", 33.35640951981520)):
        assert getattr(C, name) == pytest.approx(value, rel=1e-13), name


def test_no_hard_coded_bohr_or_hartree_factors():
    """MACE's own Python imports its Bohr, Hartree, kJ/mol and cm-1 factors from
    mace/constants.py. Allowed: mace/constants.py, the standalone fallbacks
    (``except ImportError``, pinned above) and CRYSTAL's own output text
    (``BOHR = 0.5291772083 ANGSTROM`` in a dummy output)."""
    import re
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    lit = re.compile(r"0\.529177\d*|27\.2113862\d*|219474\.63\d*|4\.359744722\d*e-18|2625\.499\d*|627\.509\d*")
    bad = []
    for p in sorted((repo / "mace").rglob("*.py")):
        rel = p.relative_to(repo).as_posix()
        if rel == "mace/constants.py":
            continue
        lines = p.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines):
            if not lit.search(line) or "BOHR = 0.5291772083 ANGSTROM" in line or "cm-1 / 219474.63" in line:
                continue
            if any("except ImportError" in lines[k] for k in range(max(0, i - 3), i)):
                continue
            bad.append(f"{rel}:{i + 1}: {line.strip()}")
    assert not bad, "\n".join(bad)


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
