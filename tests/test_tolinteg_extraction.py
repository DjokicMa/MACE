"""Regression tests: TOLINTEG extraction from real CRYSTAL outputs.

CRYSTAL prints per-tolerance exponents as "10**  -7"; pure-DFT runs print
"10**   20" for T3-T5 (exchange screening disabled). Two historical bugs:
  1. abs()'ing everything re-emitted absurd "7 7 20 20 20" input decks;
  2. the first fix skipped extraction entirely whenever ANY slot was positive,
     silently reverting a custom T1/T2 (e.g. "9 9 ...") to the full default.
Now: real (negative) exponents are kept and only disabled (positive) slots get
their per-position default — so "-9 -9 20 20 20" regenerates as "9 9 7 7 14".

Driven through CrystalOutputParser._extract_tolerances on lines from REAL
outputs (BAND-type outputs lack a final geometry so full parse() is not
applicable). The custom-T1/T2 case substitutes only the digit value on the
real T1/T2 lines (same precedent as the SPINLOCK round-trip tests: values
varied, format untouched).
"""
from pathlib import Path

from d12_parsers import CrystalOutputParser

from conftest import find_data


def _tolinteg_from_lines(source: Path, lines) -> str:
    p = CrystalOutputParser(str(source))
    p._extract_tolerances(list(lines))
    return p.data.get("tolerances", {}).get("TOLINTEG")


def _real_lines(path: Path):
    return Path(path).read_text(errors="ignore").splitlines()


def test_pure_dft_disabled_slots_get_defaults():
    """Real pure-DFT OPT output (-7 -7 20 20 20): the disabled T3-T5 must not
    surface as '20 20 20' nor kill the extraction; per-position defaults."""
    f = find_data("OPT/*BULK_OPTGEOM_TZ.out", must_contain="(T3) 10**   20")
    assert _tolinteg_from_lines(f, _real_lines(f)) == "7 7 7 7 14"


def test_custom_tolerances_survive_disabled_slots():
    """Custom T1/T2 (-9 -9) + disabled T3-T5: the user's tolerances must be
    preserved, defaulting only the disabled slots (was: whole quintet reset)."""
    f = find_data("OPT/*BULK_OPTGEOM_TZ.out", must_contain="(T3) 10**   20")
    lines = []
    for line in _real_lines(f):
        if "(T1) 10**" in line or "(T2) 10**" in line:
            line = line.replace("10**   -7", "10**   -9")
        lines.append(line)
    assert _tolinteg_from_lines(f, lines) == "9 9 7 7 14"


def test_hybrid_all_negative_unchanged():
    """Real hybrid output (-8 -8 -8 -9 -24): all-negative path is untouched."""
    f = find_data("BAND/*B3LYP*_band.out", must_contain="(T5) 10**  -24")
    assert _tolinteg_from_lines(f, _real_lines(f)) == "8 8 8 9 24"
