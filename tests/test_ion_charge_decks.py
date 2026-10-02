"""The charge planner (Crystal_d12/ion_charges.py): which records a charged system gets.

Decision table (manual p.80 CHEMOD/CHARGED, p.103 DOPING; measured on HPCC):
internal BASISSET -> DOPING -q, CHEMOD only shapes the guess; explicit basis
with CHEMOD -> CHEMOD + CHARGED and no DOPING (CHEMOD + DOPING counted the
charge twice); explicit basis without CHEMOD -> DOPING -q; q = 0 -> CHEMOD only.
The geometries are synthetic and shuffled (tests/ion_fixtures.py).
"""
from decimal import Decimal

import pytest

from conftest import REPO_ROOT
from d12_constants import read_basis_file
from ion_charges import (BasisInfo, IonChargeError, MethodInfo, SHELL_TABLE, apply_charge_setup,
                         fmt_charge, parse_basis_blocks, plan_charge_setup, strip_charge_records)
from ion_fixtures import (Z, acetate, carbonate, ec, fsi, hydroxide, li_near, molecule_deck, no3, shuffled,
                          sulfate)
from mace.utils.ion_fragments import cell_matrix

LI_PLUS = "2.0 0.0 0.0 0.0 0.0"
FSI_O = "2.0 2.0 0.0 0.0 4.25 0.0 0.0"
TZ = REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta"
INTERNAL = BasisInfo(internal="SOLDEF2MSVP")


def _split(atoms):
    syms = [s for s, _ in atoms]
    return syms, [Z[s] for s in syms], [p for _, p in atoms]


def _plan(atoms, basis=INTERNAL, method=None, **kw):
    syms, z, xyz = _split(atoms)
    kw.setdefault("ion_charges", "auto")
    return plan_charge_setup(syms, z, xyz, basis=basis, method=method or MethodInfo(), **kw)


def li_with(n_fsi, n_ec):
    parts, li = [], None
    for k in range(n_fsi):
        anion = [(s, (x + 9.0 * k, y, z)) for s, (x, y, z) in fsi()]
        parts += anion
        if li is None:
            li = li_near(anion, 2)
    for k in range(n_ec):
        mol = [(s, (x + 9.0 * k, y + 9.0, z)) for s, (x, y, z) in ec()]
        parts += mol
        if li is None:
            li = li_near(mol, 1)
    return shuffled(parts + [li], seed=n_fsi * 10 + n_ec)


# --------------------------------------------------------------- formatting

@pytest.mark.parametrize("value,text", [(2, "2.0"), (Decimal("4.25"), "4.25"), (0, "0.0"),
                                        (Decimal(13) / 3, "4.333333"), (Decimal("-1"), "-1.0")])
def test_shell_charges_have_one_to_six_decimals(value, text):
    assert fmt_charge(value) == text


def test_shell_table_has_the_group_vectors():
    assert SHELL_TABLE[("SOLDEF2MSVP", "Li")][0] == (2, 1, 0, 0, 0)
    assert SHELL_TABLE[("SOLDEF2MSVP", "O")][0] == (2, 2, 0, 0, 4, 0, 0)
    for (lib, el), (vec, types) in SHELL_TABLE.items():
        assert len(vec) == len(types), (lib, el)
        assert sum(vec) == Z.get(el, sum(vec)), (lib, el)


# ------------------------------------------------- internal library (group)

@pytest.mark.parametrize("n_fsi,n_ec,q,doping", [(1, 1, 0, None), (0, 2, 1, "-1"), (2, 0, -1, "1"),
                                                 (3, 0, -2, "2")])
def test_net_charge_gives_doping_and_the_group_vectors(n_fsi, n_ec, q, doping):
    atoms = li_with(n_fsi, n_ec)
    setup = _plan(atoms)
    assert setup.net_charge == q
    assert (setup.doping_lines() or [None, None])[1] == doping
    assert not setup.charged
    syms = [s for s, _ in atoms]
    lines = setup.chemod_lines()
    assert lines[:2] == ["CHEMOD", str(1 + 4 * n_fsi)]
    rows = {int(r.split()[0]): r.split(None, 1)[1] for r in lines[2:]}
    assert [rows[i] for i in rows if syms[i - 1] == "Li"] == [LI_PLUS]
    assert [rows[i] for i in rows if syms[i - 1] == "O"] == [FSI_O] * (4 * n_fsi)
    # the deck: CHEMOD after the BASISSET name, DOPING before the last END
    deck = apply_charge_setup(molecule_deck(atoms), setup).splitlines()
    i = deck.index("SOLDEF2MSVP")
    assert deck[i + 1] == "CHEMOD"
    if doping:
        assert deck[-3:] == ["DOPING", doping, "END"]
    else:
        assert "DOPING" not in deck


def test_q0_never_writes_doping_0():
    setup = _plan(li_with(1, 0))
    assert setup.doping is None and setup.doping_lines() == []


def test_doping_only_without_chemod():
    setup = _plan(li_with(0, 2), chemod="no")
    assert setup.chemod == [] and setup.doping_lines() == ["DOPING", "-1"]


def test_net_charge_alone_writes_doping():
    atoms = li_with(0, 2)
    setup = _plan(atoms, ion_charges=None, net_charge=1)
    assert setup.chemod == [] and setup.doping_lines() == ["DOPING", "-1"]


def test_unknown_internal_library_is_refused():
    with pytest.raises(IonChargeError, match="basis DEF2-MSVP is internal; MACE does not know Li's shell"):
        _plan(li_with(0, 2), basis=BasisInfo(internal="DEF2-MSVP"))


def test_site_states_shell_count_is_checked():
    atoms = li_with(0, 2)
    li = [s for s, _ in atoms].index("Li") + 1
    with pytest.raises(IonChargeError, match=r"3 shell charges, but its basis has 5 shells \(SSSPP\)"):
        _plan(atoms, ion_charges=None, net_charge=1,
              site_states=[{"atom": li, "shell_charges": [2, 0, 0]}])


def test_site_states_need_a_net_charge():
    with pytest.raises(IonChargeError, match="give the net charge"):
        _plan(li_with(0, 2), ion_charges=None, site_states=[{"atom": 1, "shell_charges": [2, 0, 0, 0, 0]}])


def test_site_states_override_auto():
    atoms = li_with(1, 1)
    li = [s for s, _ in atoms].index("Li") + 1
    setup = _plan(atoms, net_charge=0, site_states=[{"atom": li, "shell_charges": [2, 0.5, 0, 0, 0]}])
    rows = dict((label, vec) for label, vec in setup.chemod)
    assert [fmt_charge(v) for v in rows[li]] == ["2.0", "0.5", "0.0", "0.0", "0.0"]


def test_unknown_element_site_states_need_the_parent_out():
    atoms = [("Li", (0.0, 0.0, 0.0))]
    basis = BasisInfo(internal="STO-3G")
    with pytest.raises(IonChargeError, match="give the parent .out"):
        _plan(atoms, basis=basis, ion_charges=None, net_charge=1,
              site_states=[{"atom": 1, "shell_charges": [2, 0, 0]}])
    out = (" *******************************************************************************\n"
           " LOCAL ATOMIC FUNCTIONS BASIS SET\n"
           " *******************************************************************************\n"
           "   ATOM   X(AU)   Y(AU)   Z(AU)  N. TYPE  EXPONENT  S COEF   P COEF   D/F/G COEF\n"
           " *******************************************************************************\n"
           "   1 LI   0.000   0.000   0.000\n"
           "                                  1 S  \n"
           "                                  2 SP \n"
           " *******************************************************************************\n")
    setup = _plan(atoms, basis=basis, ion_charges=None, net_charge=1, parent_out_text=out,
                  site_states=[{"atom": 1, "shell_charges": [2, 0]}])
    assert setup.chemod_lines()[2] == "1 2.0 0.0"


def test_net_charge_differing_from_the_ions_warns():
    setup = _plan(li_with(0, 2), net_charge=0, spinlock=1)
    assert any("differs from the ions found (+1)" in w for w in setup.warnings)
    assert setup.doping is None


# ------------------------------------------- anions outside the ion table

def _salt(anion, n_li):
    """``n_li`` Li 1.95 A from the anion's O atoms (shuffled): Li2CO3, LiOH, Li2SO4, LiOAc."""
    atoms = anion()
    o_atoms = [k for k, (s, _) in enumerate(atoms) if s == "O"]
    return shuffled(atoms + [li_near(atoms, o_atoms[k]) for k in range(n_li)], seed=3)


@pytest.mark.parametrize("anion,n_li,formula,free", [(carbonate, 2, "CO3", 2), (hydroxide, 1, "HO", 1),
                                                     (sulfate, 2, "O4S", 2), (acetate, 1, "C2H3O2", 1)],
                         ids=["Li2CO3", "LiOH", "Li2SO4", "LiOAc"])
def test_auto_refuses_a_salt_whose_anion_is_not_in_the_table(anion, n_li, formula, free):
    # the review's case: auto mode counted only the Li+ and wrote DOPING -2 for Li2CO3
    with pytest.raises(IonChargeError, match=rf"count only the table ions \(\+{n_li} here\), but "
                                             rf"{formula} \({free} unpaired valence.*Give the net charge"):
        _plan(_salt(anion, n_li))


def test_a_net_charge_lets_the_salt_through_with_a_warning():
    setup = _plan(_salt(carbonate, 2), net_charge=0)
    assert setup.net_charge == 0 and setup.doping is None
    assert any("CO3 (2 unpaired valences) is not in the ion table and looks like an anion" in w
               for w in setup.warnings)


def test_an_unrecognised_anion_next_to_a_table_salt_is_refused_too():
    # LiFSI (auto net 0) with a stray carbonate: the real charge is -2
    lifsi = li_with(1, 0)
    co3 = [(s, (x, y - 9.0, z)) for s, (x, y, z) in carbonate()]
    with pytest.raises(IonChargeError, match=r"\(\+0 here\), but CO3"):
        _plan(lifsi + co3)


@pytest.mark.parametrize("n_fsi,n_ec", [(0, 2), (1, 1), (2, 0)])
def test_solvated_table_ions_are_not_flagged(n_fsi, n_ec):
    setup = _plan(li_with(n_fsi, n_ec))
    assert not any("anion" in w for w in setup.warnings)


def test_an_unjudgeable_fragment_on_a_cation_warns():
    # Fe has no VALENCES entry: MACE cannot tell whether FeO2 is an anion
    feo2 = [("Fe", (0.0, 0.0, 0.0)), ("O", (1.62, 0.0, 0.0)), ("O", (-1.62, 0.0, 0.0))]
    li = li_near(feo2, 1)
    setup = _plan(shuffled(feo2 + [li]))
    assert any("FeO2" in w and "touches a table cation" in w for w in setup.warnings)


# ----------------------------------------------------------- explicit basis

def _explicit(atoms):
    text = "".join(read_basis_file(str(TZ), z, "EXTERNAL") for z in sorted({Z[s] for s, _ in atoms}))
    return BasisInfo(blocks=parse_basis_blocks(text + "99 0\n"))


def test_explicit_basis_blocks_are_parsed():
    blocks = _explicit([("Li", (0, 0, 0)), ("O", (0, 0, 0))]).blocks
    assert blocks[3].types == ["S", "S", "S", "S", "P"]
    assert [fmt_charge(v) for v in blocks[3].che] == ["2.0", "1.0", "0.0", "0.0", "0.0"]
    assert blocks[8].valence == 8


def test_explicit_ecp_header_is_skipped():
    blocks = parse_basis_blocks((TZ / "282").read_text() + "99 0\n")
    assert blocks[282].ecp and blocks[282].valence == 22


def test_explicit_nitrate_is_chemod_plus_charged_with_exact_sums():
    atoms = shuffled(no3(), seed=3)
    setup = _plan(atoms, basis=_explicit(atoms))
    assert setup.charged and setup.doping is None
    ps = [vec[4] for label, vec in setup.chemod]
    assert [fmt_charge(p) for p in ps] == ["4.333333", "4.333333", "4.333334"]
    written = sum(Decimal(v) for line in setup.chemod_lines()[2:] for v in line.split()[1:])
    assert written == 3 * 8 + 1
    assert setup.electrons == 32
    deck = apply_charge_setup(molecule_deck(atoms, basis="END\n" + "".join(
        read_basis_file(str(TZ), z, "EXTERNAL") for z in (7, 8)) + "99 0\nEND\n"), setup).splitlines()
    i = deck.index("99 0")
    assert deck[i + 1] == "CHEMOD" and deck[i + 2] == "3" and deck[i + 6] == "CHARGED"
    assert "DOPING" not in deck


def test_explicit_neutral_ion_pair_has_no_charged():
    atoms = shuffled(no3() + [("Li", (2.6, 0.0, 0.0))], seed=5)
    setup = _plan(atoms, basis=_explicit(atoms))
    assert not setup.charged and setup.doping is None and len(setup.chemod) == 4


def test_explicit_chemod_must_remove_the_net_charge():
    atoms = no3()
    with pytest.raises(IonChargeError, match="must remove exactly the net charge"):
        _plan(atoms, basis=_explicit(atoms), net_charge=-2)


def test_explicit_without_chemod_is_doping():
    atoms = no3()
    setup = _plan(atoms, basis=_explicit(atoms), chemod="no")
    assert setup.doping_lines() == ["DOPING", "1"] and not setup.charged


def _periodic_lif(explicit):
    cell = cell_matrix(12.0, 12.0, 12.0, 90, 90, 90)
    atoms = [(s, (x + 6, y + 6, z + 6)) for s, (x, y, z) in no3()] + [("Li", (1.0, 1.0, 1.0)),
                                                                      ("Li", (11.0, 11.0, 1.0))]
    return atoms, cell, (_explicit(atoms) if explicit else INTERNAL)


def test_periodic_explicit_cell_defaults_to_doping():
    atoms, cell, basis = _periodic_lif(True)
    setup = _plan(atoms, basis=basis, cell=cell, periodic_dirs=(0, 1, 2))
    assert setup.chemod == [] and setup.doping_lines() == ["DOPING", "-1"]
    assert any("DOPING alone" in r for r in setup.report)


def test_periodic_explicit_cell_with_chemod_asked_warns_about_the_background():
    atoms, cell, basis = _periodic_lif(True)
    setup = _plan(atoms, basis=basis, cell=cell, periodic_dirs=(0, 1, 2), chemod="yes")
    assert setup.charged and setup.doping is None
    assert any("background" in w for w in setup.warnings)


def test_symmetric_periodic_chemod_is_refused_but_doping_is_allowed():
    atoms, cell, basis = _periodic_lif(False)
    with pytest.raises(IonChargeError, match="P1"):
        _plan(atoms, cell=cell, periodic_dirs=(0, 1, 2), symmetric=True)
    setup = _plan(atoms, cell=cell, periodic_dirs=(0, 1, 2), symmetric=True, chemod="no")
    assert setup.doping_lines() == ["DOPING", "-1"]


# --------------------------------------------------------------------- spin

ODD = li_with(0, 2)  # Li+ (EC)2 written as neutral: 95 electrons


def test_odd_count_without_a_spin_state_is_refused_non_interactively():
    with pytest.raises(IonChargeError, match="95 electrons is odd"):
        _plan(ODD, net_charge=0)


def test_odd_count_with_spinlock_writes_spin_and_spinlock():
    setup = _plan(ODD, net_charge=0, spinlock=1, spinlock_cycles=30)
    assert setup.spin and setup.spinlock == (1, 30)
    deck = apply_charge_setup(molecule_deck(ODD), setup).splitlines()
    assert deck[deck.index("ENDDFT") - 1] == "SPIN"
    assert deck[-3:] == ["SPINLOCK", "1 30", "END"]


def test_odd_count_asks_when_interactive():
    answers = iter(["", ""])
    setup = _plan(ODD, net_charge=0, interactive=True, ask=lambda prompt: next(answers))
    from d12_writer import DEFAULT_SPINLOCK_CYCLES
    assert setup.spinlock == (1, DEFAULT_SPINLOCK_CYCLES)


def test_odd_count_uses_the_parent_spin_state():
    setup = _plan(ODD, net_charge=0, parent_spinlock=(1, 30))
    assert setup.spinlock == (1, 30)


def test_even_spinlock_for_an_odd_count_is_refused():
    with pytest.raises(IonChargeError, match="even"):
        _plan(ODD, net_charge=0, spinlock=2)


@pytest.mark.parametrize("hf", ["RHF", "HF3C", "HFSOL3C"])
def test_closed_shell_hf_cannot_hold_an_odd_count(hf):
    with pytest.raises(IonChargeError, match="closed shell"):
        _plan(ODD, net_charge=0, spinlock=1, method=MethodInfo(method="HF", hf_method=hf))


def test_uhf_takes_spinlock_without_spin():
    setup = _plan(ODD, net_charge=0, spinlock=1, method=MethodInfo(method="HF", hf_method="UHF"))
    assert setup.spinlock == (1, 50) and not setup.spin


def test_ghosts_doping_and_an_odd_count_is_refused():
    with pytest.raises(IonChargeError, match="NALPHA-NBETA NOT COMPATIBLE"):
        _plan(li_with(0, 1) + [("Li", (30.0, 0.0, 0.0))], net_charge=1, spinlock=1,
              method=MethodInfo(ghosts=True))


# ----------------------------------------------------------------- writing

def test_a_deck_with_records_is_refused():
    deck = molecule_deck(ODD).replace("ENDDFT\n", "ENDDFT\nDOPING\n-1\n")
    with pytest.raises(IonChargeError, match="already has DOPING"):
        apply_charge_setup(deck, _plan(li_with(0, 2)))


def test_strip_removes_every_record():
    deck = molecule_deck(ODD)
    charged = apply_charge_setup(deck, _plan(li_with(0, 2)))
    assert strip_charge_records(charged) == deck


def test_a_periodic_cell_may_stay_closed_shell_with_an_odd_count():
    cell = cell_matrix(12.0, 12.0, 12.0, 90, 90, 90)
    atoms = [(s, (x + 6, y + 6, z + 6)) for s, (x, y, z) in li_with(0, 2)]
    setup = _plan(atoms, cell=cell, periodic_dirs=(0, 1, 2), net_charge=0)
    assert setup.spinlock is None and not setup.spin
    assert any("half-filled band" in w for w in setup.warnings)
    # given a spin state, it is used
    setup = _plan(atoms, cell=cell, periodic_dirs=(0, 1, 2), net_charge=0, spinlock=1)
    assert setup.spinlock == (1, 50)


def test_odd_molecule_with_a_screened_hybrid_is_written_with_a_warning():
    setup = _plan(ODD, net_charge=0, spinlock=1, method=MethodInfo(functional="HSESOL3C"))
    assert setup.spinlock == (1, 50)
    assert any("HSESOL3C as asked" in w for w in setup.warnings)
    quiet = _plan(ODD, net_charge=0, spinlock=1, method=MethodInfo(functional="PBE0"))
    assert not any("screened" in w for w in quiet.warnings)


def test_deck_method_reads_the_functional():
    from ion_charges import deck_method
    assert deck_method(molecule_deck(ODD)).functional == "HSESOL3C"
    assert deck_method(molecule_deck(ODD, spin=True, dft="PBE0\nXLGRID\n")).functional == "PBE0"
