"""With an internal BASISSET library the net charge is always DOPING; CHEMOD only shapes the guess.

Measured on CRYSTAL23 (HPCC): after a library name CHEMOD changes the
starting guess only, never the electron count (the guess is renormalized),
so the count has to come from DOPING (the group's 37 HSEsol-3c OPTs and 62
FREQs do exactly that). CHARGED is never needed with a library basis. This
pins the planner's answer for every way of asking: auto ions, a plain net
charge, site_states, chemod yes/auto/no, molecule or periodic cell.
"""
from decimal import Decimal

import pytest

from ion_charges import BasisInfo, MethodInfo, apply_charge_setup, plan_charge_setup
from ion_fixtures import Z, ec, fsi, li_near, molecule_deck, shuffled
from mace.utils.ion_fragments import cell_matrix

LI_PLUS = [Decimal(2), Decimal(0), Decimal(0), Decimal(0), Decimal(0)]


def _li_fsi2():
    a1, a2 = fsi(), [(s, (x + 9.0, y, z)) for s, (x, y, z) in fsi()]
    return shuffled(a1 + a2 + [li_near(a1, 2)], seed=5)  # Li+ (FSI-)2: q = -1


def _li_ec2():
    m1, m2 = ec(), [(s, (x + 9.0, y + 9.0, z)) for s, (x, y, z) in ec()]
    return shuffled(m1 + m2 + [li_near(m1, 1)], seed=6)  # Li+ (EC)2: q = +1


def _plan(atoms, periodic=False, **kw):
    syms = [s for s, _ in atoms]
    xyz = [p for _, p in atoms]
    if periodic:
        xyz = [(x + 15, y + 15, z + 15) for x, y, z in xyz]
        kw.update(cell=cell_matrix(30.0, 30.0, 30.0, 90, 90, 90), periodic_dirs=(0, 1, 2))
    return plan_charge_setup(syms, [Z[s] for s in syms], xyz, basis=BasisInfo(internal="SOLDEF2MSVP"),
                             method=MethodInfo(functional="HSESOL3C"), **kw)


@pytest.mark.parametrize("periodic", [False, True], ids=["MOLECULE", "CRYSTAL"])
@pytest.mark.parametrize("chemod", ["auto", "yes", "no"])
@pytest.mark.parametrize("atoms,q", [(_li_ec2(), 1), (_li_fsi2(), -1)], ids=["q+1", "q-1"])
def test_auto_ions_charge_is_doping_and_never_charged(atoms, q, chemod, periodic):
    setup = _plan(atoms, periodic=periodic, ion_charges="auto", chemod=chemod)
    assert setup.doping == -q
    assert setup.charged is False
    assert bool(setup.chemod) == (chemod != "no")
    neutral = sum(Z[s] for s, _ in atoms)
    assert setup.electrons == neutral - q


@pytest.mark.parametrize("q", [1, -1, 3, Decimal("0.5")])  # Li(EC)2 neutral is 95
def test_a_plain_net_charge_is_doping_alone(q):
    setup = _plan(_li_ec2(), net_charge=q)
    assert setup.doping == -Decimal(str(q)) and not setup.chemod and not setup.charged


def test_site_states_shape_the_guess_and_doping_sets_the_count():
    atoms = _li_ec2()
    li = [s for s, _ in atoms].index("Li") + 1
    setup = _plan(atoms, net_charge=1, site_states=[{"atom": li, "shell_charges": [2, 0, 0, 0, 0]}])
    assert setup.chemod == [(li, LI_PLUS)]
    assert setup.doping == -1 and not setup.charged


def test_neutral_ion_pair_gets_chemod_and_no_doping():
    a = fsi()
    atoms = shuffled(a + [li_near(a, 2)], seed=7)
    setup = _plan(atoms, ion_charges="auto")
    assert setup.chemod and setup.doping is None and not setup.charged


def test_the_written_deck_has_doping_before_the_final_end_and_no_charged():
    atoms = _li_ec2()
    deck = apply_charge_setup(molecule_deck(atoms), _plan(atoms, ion_charges="auto")).splitlines()
    assert "CHARGED" not in deck
    assert deck[deck.index("SOLDEF2MSVP") + 1] == "CHEMOD"
    assert deck[-3:] == ["DOPING", "-1", "END"]
