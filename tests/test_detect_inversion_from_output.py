"""detect_inversion_from_crystal_output: a 2-fold axis is not an inversion.

The SeeK-path fallback (no seekpath library) reads inversion from the CRYSTAL
output and gives a non-centrosymmetric structure the ``<variant>_noinv`` path.
Two of its three methods said "centrosymmetric" when the structure was not:

* the SYMMOPS check matched the text of the first two matrix rows,
  "-1.00 0.00 0.00 0.00 -1.00 0.00", which a 2-fold axis along z shares with
  the inversion; every non-centrosymmetric group with that axis (222, mm2,
  4, -4, 422, 4mm, -42m, 6, 622, 6mm, 23, 432, -43m, ...) was called
  centrosymmetric;
* the space-group-number check took a digit out of a symbol, "P N N 2" -> 2.

The SYMMOPS rows below keep the layout of tests/data/opt_restart/tqb_pto.out
(a real CRYSTAL23 output); the C2v operators are the urea molecule's from the
CRYSTAL23 manual, p. 387.
"""
import sys
from pathlib import Path

import pytest

_D3 = str(Path(__file__).resolve().parent.parent / "Crystal_d3")
if _D3 not in sys.path:
    sys.path.insert(0, _D3)

from d3_kpoints import detect_inversion_from_crystal_output  # noqa: E402

SYMMOPS_HEAD = """\
 T = ATOM BELONGING TO THE ASYMMETRIC UNIT

 ****    4 SYMMOPS - TRANSLATORS IN FRACTIONAL UNITS
 **** MATRICES AND TRANSLATORS IN THE CRYSTALLOGRAPHIC REFERENCE FRAME
   V INV                    ROTATION MATRICES                   TRANSLATORS
"""

TAIL = """
 DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)
          X                    Y                    Z
"""

# C2v: identity, 2-fold along z, two mirrors (manual p. 387).
C2V_OPS = """\
   1   1  1.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00
   2   2 -1.00  0.00  0.00  0.00 -1.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00
   3   3 -1.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00
   4   4  1.00  0.00  0.00  0.00 -1.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00
"""

# C2h: identity, 2-fold along z, inversion, mirror perpendicular to z. The
# inversion carries a translator (an inversion centre away from the origin)
# and prints "-0.00", as tests/data/opt_restart/tqc_agbr.out does.
C2H_OPS = """\
   1   1  1.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00
   2   2 -1.00  0.00  0.00  0.00 -1.00  0.00  0.00  0.00  1.00  0.00  0.00  0.50
   3   3 -1.00 -0.00 -0.00 -0.00 -1.00 -0.00  0.00  0.00 -1.00  0.50  0.00  0.00
   4   4  1.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00 -1.00  0.50  0.00  0.50
"""


def detect(tmp_path, text):
    out = tmp_path / "x.out"
    out.write_text(text)
    return detect_inversion_from_crystal_output(str(out))


def test_two_fold_axis_along_z_is_not_an_inversion(tmp_path):
    assert detect(tmp_path, SYMMOPS_HEAD + C2V_OPS + TAIL) == (
        False, "no_inversion_operator")


def test_inversion_with_a_translator_is_an_inversion(tmp_path):
    assert detect(tmp_path, SYMMOPS_HEAD + C2H_OPS + TAIL) == (
        True, "inversion_operator")


def test_real_centrosymmetric_output(tmp_path):
    """tqb_pto.out: Pm-3m, found by its SPACE GROUP line."""
    real = Path(__file__).parent / "data" / "opt_restart" / "tqb_pto.out"
    assert detect_inversion_from_crystal_output(str(real)) == (
        True, "explicit_centrosymmetric")


def test_real_symmops_table_holds_an_inversion(tmp_path):
    """The Pm-3m SYMMOPS table of tqb_pto.out alone (no SPACE GROUP line)."""
    real = Path(__file__).parent / "data" / "opt_restart" / "tqb_pto.out"
    text = real.read_text()
    start = text.index(" T = ATOM BELONGING TO THE ASYMMETRIC UNIT")
    end = text.index(" DIRECT LATTICE VECTORS CARTESIAN COMPONENTS", start)
    assert detect(tmp_path, text[start:end + 60]) == (
        True, "inversion_operator")


# The 3D space-group line's wording for a non-centrosymmetric group is
# assumed here by analogy with the real "SPACE GROUP (CENTROSYMMETRIC)" line;
# the check does not depend on the words in the parentheses.
PNN2_LINE = " SPACE GROUP (NONCENTROSYMMETRIC)     :  P N N 2\n"


def test_digit_in_a_space_group_symbol_is_not_a_number(tmp_path):
    assert detect(tmp_path, PNN2_LINE) == (False, "unknown")


@pytest.mark.parametrize("number, centro", [(47, True), (25, False), (191, True)])
def test_corresponding_space_group_number(tmp_path, number, centro):
    """A slab output prints the 3D group's number (graphene_lg37.out)."""
    text = (
        " TWO-SIDED PLANE GROUP N. 37          :  P M M M\n"
        f" CORRESPONDING SPACE GROUP N. {number:3d}\n"
    )
    assert detect(tmp_path, text) == (centro, "space_group_number")
