"""`mace phonopy generate`: reading a parent deck into a supercell template.

No phonopy needed. The template is what phonopy's CRYSTAL interface calls
TEMPLATE: the parent's basis set and block 3, with no final END, no TOLDEE and
no GRADCAL. The deck is ``EXTERNAL`` (manual p.21) - BASISSET replaces the END
of block 1 (p.28) - then the template, then ``TOLDEE / N / GRADCAL / END``, the
tail phonopy's own writer appends. Edits: (a) final END removed, (b) TOLDEE
moved to the tail and raised to at least 10, (c) GUESSP dropped, (d) SHRINK
divided by the supercell size, (e) BIPOSIZE/EXCHSIZE added under d12_writer's
rule, (f) GRADCAL moved to the tail. The method, basis and tolerances must come
through byte for byte: a phonon run on a different method than the parent's is
a different calculation.

Measured on CRYSTAL23 (phonopy 4.7.2): a TEMPLATE that ends with END gives
``... END / TOLDEE 10 / GRADCAL / END``; CRYSTAL stops at the first END, the
SCF finishes with return code 0 and no forces are written.
"""
from pathlib import Path
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
    assert P.parent_toldee(d.block3) == 7


def test_hsesol3c_soldef2msvp_survives_exactly():
    text = internal(label="SOLDEF2MSVP", method="HSESOL3C")
    d = P.split_parent_deck(text)
    t = P.template_lines(d, n_super_atoms=16)
    assert t.lines[:2] == ["BASISSET", "SOLDEF2MSVP"]
    assert t.lines[2:6] == ["DFT", "HSESOL3C", "XLGRID", "ENDDFT"]


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
    t = P.template_lines(d, n_super_atoms=16)
    body = t.lines[len(d.basis_lines):]
    assert body[:11] == _lines(NESTED_B3)[:11]          # DFT, DFTD3, GCP verbatim
    assert "GUESSP" not in body
    assert "GRADCAL" not in body and body[-1] != "END"
    assert P.deck_text("t", d, t).splitlines()[-4:] == ["TOLDEE", "11", "GRADCAL", "END"]


def test_deck_without_final_end_is_refused():
    with pytest.raises(P.DeckSplitError):
        P.split_parent_deck(internal().rstrip("\n").rsplit("\n", 1)[0] + "\n")


# ---------------------------------------------------------------------------
# T2: template edits
# ---------------------------------------------------------------------------

def _toldees(lines):
    return [lines[i + 1] for i, ln in enumerate(lines) if ln == "TOLDEE"]


def _tail(d, t):
    return P.deck_text("t", d, t).splitlines()[-4:]


def test_toldee_moved_to_the_tail_and_raised():
    d = P.split_parent_deck(internal("7"))
    t = P.template_lines(d, 16)
    assert _toldees(t.lines) == []                    # not in the template
    assert t.toldee == 10 and _tail(d, t) == ["TOLDEE", "10", "GRADCAL", "END"]
    assert any(e.startswith("(b) TOLDEE 7 -> 10") for e in t.edits)


def test_toldee_kept_when_already_tighter():
    d = P.split_parent_deck(internal("11"))
    t = P.template_lines(d, 16)
    assert _toldees(t.lines) == [] and _tail(d, t)[:2] == ["TOLDEE", "11"]


def test_toldee_requested_can_only_raise():
    d = P.split_parent_deck(internal("11"))
    assert P.template_lines(d, 16, toldee=8).toldee == 11
    assert P.template_lines(d, 16, toldee=12).toldee == 12


def test_toldee_added_when_absent():
    text = internal().replace("TOLDEE\n7\nSHRINK", "SHRINK")  # the OPTGEOM TOLDEE stays in block 1
    d = P.split_parent_deck(text)
    t = P.template_lines(d, 16)
    assert _toldees(t.lines) == [] and _tail(d, t) == ["TOLDEE", "10", "GRADCAL", "END"]
    deck = P.deck_text("t", d, t).splitlines()
    assert deck.count("TOLDEE") == 1


def _block3_part(d, t):
    return t.lines[len(d.basis_lines):]


@pytest.mark.parametrize("text", [internal(), internal(extra_b3="GRADCAL\n"), explicit(),
                                  explicit(NESTED_B3)],
                         ids=["internal", "internal-gradcal", "explicit", "explicit-nested"])
def test_template_has_no_end_toldee_or_gradcal(text):
    """phonopy appends TOLDEE / 10 / GRADCAL / END to the TEMPLATE; an END left
    in it stops CRYSTAL before GRADCAL (rc 0, no forces)."""
    d = P.split_parent_deck(text)
    t = P.template_lines(d, 16)
    body = _block3_part(d, t)
    top = [k for _, k in P._top_level(body)]
    assert "END" not in top and body[-1].strip().upper() != "END"
    assert "TOLDEE" not in top and "GRADCAL" not in top
    for banned in ("OPTGEOM", "FREQCALC", "ENDOPT", "ENDFREQ", "CRYSTAL"):
        assert banned not in [ln.strip() for ln in t.lines]
    tmpl = P.template_text(t)
    assert not tmpl.rstrip().endswith("\nEND")
    deck = P.deck_text("t", d, t).splitlines()
    assert deck[-4:] == ["TOLDEE", str(t.toldee), "GRADCAL", "END"]
    assert [ln.strip() for ln in deck].count("GRADCAL") == 1
    assert [ln.strip() for ln in deck].count("TOLDEE") == 1


def test_trailing_end_in_a_template_is_caught():
    """The check the tests above rely on really fails on the broken layout."""
    d = P.split_parent_deck(internal())
    t = P.template_lines(d, 16)
    broken = P.Template(lines=t.lines + ["END"], edits=t.edits, toldee=t.toldee)
    assert [k for _, k in P._top_level(_block3_part(d, broken))][-1] == "END"
    with pytest.raises(ValueError):
        P.deck_text("t", d, broken)


def test_every_top_level_toldee_leaves_the_template():
    """Two TOLDEE records in block 3 (CRYSTAL reads both): neither may stay in
    the TEMPLATE, and the tail takes the tighter one."""
    d = P.split_parent_deck(internal("7", extra_b3="TOLDEE\n12\n"))
    t = P.template_lines(d, 16)
    assert _toldees(t.lines) == [] and t.toldee == 12
    assert _tail(d, t) == ["TOLDEE", "12", "GRADCAL", "END"]


def test_guessp_dropped():
    t = P.template_lines(P.split_parent_deck(internal(extra_b3="GUESSP\n")), 16)
    assert "GUESSP" not in t.lines
    assert any(e.startswith("(c)") for e in t.edits)


def test_shrink_verbatim_without_dims():
    d = P.split_parent_deck(internal())
    t = P.template_lines(d, 16)
    i = t.lines.index("SHRINK")
    assert t.lines[i + 1] == "8 16"
    assert not any(e.startswith("(d)") for e in t.edits)


def test_shrink_scaled_simple_form():
    t = P.template_lines(P.split_parent_deck(internal()), 16, scale_shrink_dims=[2, 2, 2])
    i = t.lines.index("SHRINK")
    assert t.lines[i + 1] == "4 8"
    note = [e for e in t.edits if e.startswith("(d)")][0]
    assert "does not divide" not in note


def test_shrink_scaled_directional_form_and_inexact_note():
    d = P.split_parent_deck(explicit(NESTED_B3))
    t = P.template_lines(d, 16, scale_shrink_dims=[2, 2, 4])
    i = t.lines.index("SHRINK")
    assert t.lines[i + 1:i + 3] == ["0 8", "4 4 2"]
    t = P.template_lines(d, 16, scale_shrink_dims=[3, 3, 3])
    i = t.lines.index("SHRINK")
    assert t.lines[i + 1:i + 3] == ["0 6", "3 3 2"]
    note = [e for e in t.edits if e.startswith("(d)")][0]
    assert "does not divide" in note and "8/3" in note


def test_simple_shrink_with_unequal_dims_becomes_directional():
    t = P.template_lines(P.split_parent_deck(internal()), 16, scale_shrink_dims=[1, 1, 2])
    i = t.lines.index("SHRINK")
    assert t.lines[i + 1:i + 3] == ["0 16", "8 8 4"]


def test_buffers_added_only_for_large_supercells_without_parent_buffers():
    d = P.split_parent_deck(internal())
    t = P.template_lines(d, 16)
    assert t.lines.count("BIPOSIZE") == 1 and t.lines.count("EXCHSIZE") == 1
    assert t.lines[t.lines.index("BIPOSIZE") + 1] == P.BUFFER_SIZE
    t = P.template_lines(d, 5)
    assert "BIPOSIZE" not in t.lines
    d = P.split_parent_deck(internal(extra_b3="BIPOSIZE\n30000000\n"))
    t = P.template_lines(d, 128)
    assert t.lines.count("BIPOSIZE") == 1 and "EXCHSIZE" not in t.lines
    assert t.lines[t.lines.index("BIPOSIZE") + 1] == "30000000"


def test_parent_gradcal_moves_to_the_tail():
    d = P.split_parent_deck(internal(extra_b3="GRADCAL\n"))
    t = P.template_lines(d, 16)
    assert "GRADCAL" not in t.lines
    deck = P.deck_text("t", d, t).splitlines()
    assert deck.count("GRADCAL") == 1 and deck[-2:] == ["GRADCAL", "END"]


@pytest.mark.parametrize("text", [internal(), internal(label="SOLDEF2MSVP", method="HSESOL3C"),
                                  explicit(), explicit(NESTED_B3)])
def test_method_and_basis_lines_byte_identical(text):
    d = P.split_parent_deck(text)
    t = P.template_lines(d, 16, scale_shrink_dims=[2, 2, 2])
    deck = P.deck_text("t", d, t)
    again = P.split_parent_deck(deck)
    assert P.method_lines(again) == P.method_lines(d)
    # Every parent record survives except the edited TOLDEE/SHRINK values and GUESSP.
    parent_tolinteg = d.block3[d.block3.index("TOLINTEG") + 1]
    assert again.block3[again.block3.index("TOLINTEG") + 1] == parent_tolinteg


def test_deck_layout_internal_and_explicit():
    d = P.split_parent_deck(internal())
    lines = P.deck_text("title", d, P.template_lines(d, 16)).splitlines()
    assert lines[:4] == ["title", "EXTERNAL", "BASISSET", "POB-TZVP-REV2"]
    d = P.split_parent_deck(explicit())
    lines = P.deck_text("title", d, P.template_lines(d, 16)).splitlines()
    assert lines[:4] == ["title", "EXTERNAL", "END", "12 4"]
    assert lines[-4:] == ["TOLDEE", "10", "GRADCAL", "END"]


# Measured on CRYSTAL23 with phonopy 4.7.2 (diamond, B3LYP-D3, parent
# test/SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12): this TEMPLATE block 3,
# after the parent's basis verbatim, ran and gave forces for 2x2x2 (SHRINK 6 12)
# and 3x3x3 (SHRINK 4 8). The parent has TOLDEE 9 and SHRINK 12 24.
DIAMOND_BLOCK3 = ["END", "DFT", "SPIN", "B3LYP-D3", "XLGRID", "ENDDFT", "TOLINTEG",
                  "8 8 8 9 24", "SHRINK", "{shrink}", "SCFDIR", "MAXCYCLE", "1600", "FMIXING",
                  "30", "DIIS", "HISTDIIS", "100", "PPAN"]


@pytest.mark.parametrize("n,shrink", [(2, "6 12"), (3, "4 8")])
def test_diamond_template_matches_the_measured_one(n, shrink):
    from conftest import find_data
    parent = find_data("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12")
    text = Path(parent).read_text()
    d = P.split_parent_deck(text)
    t = P.template_lines(d, 2 * n ** 3, scale_shrink_dims=[n, n, n])
    want_basis = text.splitlines()[text.splitlines().index("END") + 1:]
    want_basis = want_basis[:want_basis.index("99 0") + 1]
    want = want_basis + [ln.format(shrink=shrink) for ln in DIAMOND_BLOCK3]
    # MACE adds the buffers d12_writer adds above 5 atoms; CRYSTAL23 accepts them.
    got = [ln for ln in t.lines
           if ln not in ("BIPOSIZE", "EXCHSIZE", P.BUFFER_SIZE)]
    assert got == want


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
    # A closed-shell spin-polarised run (all spins ~0) is fine.
    out = GOOD_OUT + MULLIKEN.format(total="0.000000", s1="0.000", s2="-0.000")
    assert _codes(internal(), out) == []


def test_sp_parent_warns():
    text = internal().replace(OPTGEOM, "")
    out = GOOD_OUT.replace(" * OPT END - CONVERGED * E(AU):  -2.753389115686E+02  POINTS    4 *\n", "")
    d = P.split_parent_deck(text)
    assert P.refusal_reasons(d, out) == []
    assert [c for c, _ in P.warnings_for(d, out)] == [P.WARN_SP_PARENT]
