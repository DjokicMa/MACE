"""Reading a parent deck for phonopy's displaced supercells (opt2d12's PHONOPY set).

No phonopy needed. The parent deck is cut at its basis set (block 1, the
basis, block 3) to find what does not carry over to a P1 supercell (the
refusals), and the supercell's SHRINK is the parent's divided by the supercell
size. The decks themselves are opt2d12's (tests/test_phonopy_generate.py).
"""
import pytest

from mace.phonopy_bridge import parent as P

# ---------------------------------------------------------------------------
# Synthetic decks
# ---------------------------------------------------------------------------

GEOM = """CRYSTAL
0 0 0
225
4.21
2
12 0.0 0.0 0.0
8 0.5 0.5 0.5
"""

OPTGEOM = """OPTGEOM
FULLOPTG
MAXCYCLE
800
TOLDEE
7
ENDOPT
"""

BLOCK3 = """DFT
PBE0
XLGRID
ENDDFT
TOLINTEG
7 7 7 7 14
TOLDEE
{toldee}
SHRINK
8 16
SCFDIR
MAXCYCLE
800
FMIXING
30
DIIS
HISTDIIS
100
PPAN
END
"""


def internal(toldee="7", extra_b3="", label="POB-TZVP-REV2", method="PBE0"):
    b3 = BLOCK3.format(toldee=toldee).replace("PBE0", method)
    b3 = b3.replace("PPAN\n", extra_b3 + "PPAN\n")
    return "mgo_opt\n" + GEOM + OPTGEOM + f"BASISSET\n{label}\n" + b3


ECP_BASIS = """12 4
0 0 6 2.0 1.0
  11.0699 0.5432
  2.0 0.2
  1.0 0.1
  0.5 0.05
  0.2 0.02
  0.1 0.01
0 1 3 8.0 1.0
  99 0.25 0.5
  1.0 0.2 0.3
  0.3 0.1 0.2
0 1 1 0.0 1.0
  0.1 1.0 1.0
0 3 1 0.0 1.0
  0.3 1.0
282 3
INPUT
22. 0 2 2 0 0 0
12.0699 0.5432 -1
4.99 0.123 0
3.1 99.0 0
2.2 1.1 0
0 0 1 2.0 1.0
  1.0 1.0
0 1 1 2.0 1.0
  0.5 1.0 1.0
8 2
0 0 1 2.0 1.0
  1.0 1.0
0 1 1 6.0 1.0
  0.5 1.0 1.0
99 0
END
"""


def explicit(block3=None):
    b3 = block3 if block3 is not None else BLOCK3.format(toldee="7")
    return "mgo_opt\n" + GEOM + OPTGEOM + "END\n" + ECP_BASIS + b3


NESTED_B3 = """DFT
PBE0
ENDDFT
DFTD3
VERSION
4
END
GCP
METHOD
DFT/SVP
END
TOLINTEG
8 8 8 8 16
TOLDEE
11
SHRINK
0 16
 8 8 6
GUESSP
MAXCYCLE
100
END
"""


def _lines(text):
    return text.splitlines()


# ---------------------------------------------------------------------------
# T2: splitting
# ---------------------------------------------------------------------------

def test_internal_basis_with_optgeom_endopt():
    d = P.split_parent_deck(internal())
    assert d.basis_kind == "internal"
    assert d.basis_lines == ["BASISSET", "POB-TZVP-REV2"]
    assert d.block3[0] == "DFT" and d.block3[-1] == "PPAN"      # final END stripped
    assert d.dimension_keyword == "CRYSTAL"
    assert d.dropped_keywords == ["OPTGEOM"]                     # its TOLDEE is not block 3
    assert d.block3[d.block3.index("TOLDEE") + 1] == "7"


def test_hsesol3c_soldef2msvp_survives_exactly():
    text = internal(label="SOLDEF2MSVP", method="HSESOL3C")
    d = P.split_parent_deck(text)
    assert d.basis_lines == ["BASISSET", "SOLDEF2MSVP"]
    assert d.block3[:4] == ["DFT", "HSESOL3C", "XLGRID", "ENDDFT"]
    assert P.method_lines(d) == {"basis": ["BASISSET", "SOLDEF2MSVP"], "DFT": ["DFT", "HSESOL3C", "XLGRID", "ENDDFT"]}


def test_explicit_basis_is_cut_at_the_whole_line_99_0_only():
    """'99 0' inside an exponent or ECP record ("11.0699 0.5432", "4.99 0.123 0")
    is not the end of the basis - matching it as a substring was the INPBAS
    truncation bug."""
    d = P.split_parent_deck(explicit())
    assert d.basis_kind == "explicit"
    assert d.basis_lines == _lines(ECP_BASIS)
    assert d.block3[0] == "DFT"


def test_nested_dft_dftd3_gcp_ends_do_not_end_block3():
    d = P.split_parent_deck(explicit(NESTED_B3))
    assert d.block3 == _lines(NESTED_B3)[:-1]
    assert P.unknown_block3_keywords(d.block3) == []   # VERSION/METHOD are inside sub-blocks
    assert d.block3[:11] == _lines(NESTED_B3)[:11]      # DFT, DFTD3, GCP whole
    assert P.method_lines(d)["DFTD3"][-1].strip().upper().startswith("END")


def test_deck_without_final_end_is_refused():
    with pytest.raises(P.DeckSplitError):
        P.split_parent_deck(internal().rstrip("\n").rsplit("\n", 1)[0] + "\n")


# ---------------------------------------------------------------------------
# T2b: the supercell's SHRINK (opt2d12 writes the decks: d12_calc_phonopy)
# ---------------------------------------------------------------------------

def test_shrink_scaled_simple_form():
    new, isp, note = P.scaled_mesh([8, 8, 8], 16, [2, 2, 2])
    assert (new, isp) == ([4, 4, 4], 8)
    assert note == "SHRINK 8 8 8 (ISP 16) -> 4 4 4 (ISP 8)"


def test_shrink_scaled_directional_form_and_inexact_note():
    assert P.scaled_mesh([8, 8, 6], 16, [2, 2, 4])[:2] == ([4, 4, 2], 8)
    new, isp, note = P.scaled_mesh([8, 8, 6], 16, [3, 3, 3])
    assert (new, isp) == ([3, 3, 2], 6)
    assert "does not divide" in note and "8/3" in note


def test_simple_shrink_with_unequal_dims_becomes_directional():
    import d12_calc_phonopy as PH
    assert P.scaled_mesh([8, 8, 8], 16, [1, 1, 2])[:2] == ([8, 8, 4], 16)
    assert PH.supercell_k_points("8 16", [1, 1, 2])[0] == (8, 8, 4)
    assert PH.supercell_k_points("8 16", [2, 2, 2])[0] == "4 8"
    assert PH.supercell_k_points((12, 12, 6), [2, 2, 2])[0] == (6, 6, 3)
    assert PH.supercell_k_points(None, [2, 2, 2]) is None


# ---------------------------------------------------------------------------
# T3: refusals
# ---------------------------------------------------------------------------

GOOD_OUT = """ CRYSTAL CALCULATION
 == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -2.75E+02 CYCLES   9
 * OPT END - CONVERGED * E(AU):  -2.753389115686E+02  POINTS    4 *
 EEEEEEEEEE TERMINATION  DATE 01 10 2026 TIME 22:02:04.9
"""


def _codes(text, out=GOOD_OUT):
    return [c for c, _ in P.refusal_reasons(P.split_parent_deck(text), out)]


def test_good_parent_has_no_refusal():
    assert _codes(internal()) == []
    assert _codes(explicit(NESTED_B3)) == []


@pytest.mark.parametrize("dim_kw", ["SLAB", "POLYMER", "MOLECULE"])
def test_non_3d_refused(dim_kw):
    text = internal().replace("CRYSTAL\n0 0 0\n225", f"{dim_kw}\n1")
    assert P.REFUSE_NOT_3D in _codes(text)
    msg = dict(P.refusal_reasons(P.split_parent_deck(text), GOOD_OUT))[P.REFUSE_NOT_3D]
    assert "pp.73, 232" in msg
    assert P.REFUSE_NOT_3D in _codes(internal(), GOOD_OUT.replace("CRYSTAL CALC", "SLAB CALC"))


@pytest.mark.parametrize("rec", ["CHEMOD\n1\n1 2.0 8.0\n", "CHARGED\n", "DOPING\n-1\n"])
def test_charge_records_refused(rec):
    text = internal().replace("POB-TZVP-REV2\n", "POB-TZVP-REV2\n" + rec, 1) \
        if not rec.startswith("DOPING") else internal(extra_b3=rec)
    assert P.REFUSE_CHARGE_RECORDS in _codes(text)


@pytest.mark.parametrize("rec", ["ATOMSPIN\n1\n1 1\n", "SPINLOCK\n2 30\n", "BETALOCK\n2 30\n"])
def test_spin_locks_refused(rec):
    assert P.REFUSE_SPIN_LOCKS in _codes(internal(extra_b3=rec))


@pytest.mark.parametrize("rec,code", [("ATOMSPIN\n1\n1 1\n", "SPIN"), ("SPINLOCK\n2 30\n", "SPIN"),
                                      ("BETALOCK\n2 30\n", "SPIN"), ("DOPING\n-1\n", "CHARGE")])
def test_named_refusal_is_not_repeated_as_an_unknown_keyword(rec, code):
    """A SPINLOCK parent was refused twice, as SPIN_LOCKS and as UNKNOWN_KEYWORD."""
    codes = _codes(internal(extra_b3=rec))
    assert codes == [P.REFUSE_SPIN_LOCKS if code == "SPIN" else P.REFUSE_CHARGE_RECORDS]


def test_ghosts_refused():
    text = internal().replace("POB-TZVP-REV2\n", "POB-TZVP-REV2\nGHOSTS\n1\n2\n", 1)
    assert P.REFUSE_GHOSTS in _codes(text)


def test_unknown_block3_keyword_refused_by_name():
    reasons = dict(P.refusal_reasons(P.split_parent_deck(internal(extra_b3="EIGSHIFT\n1\n2 0.5\n")),
                                     GOOD_OUT))
    assert "EIGSHIFT" in reasons[P.REFUSE_UNKNOWN_KEYWORD]


def test_freq_output_as_geometry_refused():
    text = internal().replace(OPTGEOM, "FREQCALC\nNUMDERIV\n2\nENDFREQ\n")
    assert P.REFUSE_FREQ_GEOMETRY in _codes(text, GOOD_OUT.replace(" * OPT END - CONVERGED", " x"))
    out = GOOD_OUT + "                       FREQUENCY CALCULATION\n"
    assert P.REFUSE_FREQ_GEOMETRY in _codes(internal(), out)


def test_unconverged_scf_refused():
    out = GOOD_OUT.replace("CONVERGENCE ON ENERGY", "TOO MANY CYCLES")
    assert P.REFUSE_SCF_NOT_CONVERGED in _codes(internal(), out)


def test_unconverged_opt_refused():
    out = GOOD_OUT.replace(" * OPT END - CONVERGED * E(AU):  -2.753389115686E+02  POINTS    4 *\n", "")
    assert P.REFUSE_OPT_NOT_CONVERGED in _codes(internal(), out)


def test_abnormal_termination_refused():
    out = GOOD_OUT.replace(" EEEEEEEEEE TERMINATION", " ERROR **** NEIGHB **** X\n")
    assert P.REFUSE_ABNORMAL_TERMINATION in _codes(internal(), out)


MULLIKEN = """     ALPHA-BETA ELECTRONS
 MULLIKEN POPULATION ANALYSIS - NO. OF ELECTRONS    {total}

  ATOM    Z CHARGE  A.O. POPULATION

   1 MG  12  {s1}  0.000  0.000
                   0.000
   2 O    8  {s2}  0.000  0.000

  ATOM    Z CHARGE  SHELL POPULATION
"""


def test_unlocked_magnetic_parent_refused():
    out = GOOD_OUT + MULLIKEN.format(total="2.000000", s1="0.050", s2="1.950")
    reasons = dict(P.refusal_reasons(P.split_parent_deck(internal()), out))
    assert P.REFUSE_UNLOCKED_MAGNETISM in reasons
    assert "atom 2" in reasons[P.REFUSE_UNLOCKED_MAGNETISM]
    # A closed-shell spin-polarized run (all spins ~0) is fine.
    out = GOOD_OUT + MULLIKEN.format(total="0.000000", s1="0.000", s2="-0.000")
    assert _codes(internal(), out) == []


def test_sp_parent_warns():
    text = internal().replace(OPTGEOM, "")
    out = GOOD_OUT.replace(" * OPT END - CONVERGED * E(AU):  -2.753389115686E+02  POINTS    4 *\n", "")
    d = P.split_parent_deck(text)
    assert P.refusal_reasons(d, out) == []
    assert [c for c, _ in P.warnings_for(d, out)] == [P.WARN_SP_PARENT]
