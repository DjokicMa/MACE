"""Clean geometry: the spring pre-relaxation on distorted textbook cells."""
import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("spglib")

from ase.build import bulk  # noqa: E402

from mace.gui import clean as C  # noqa: E402
from mace.gui import structures as S  # noqa: E402

CELLS = {
    "Si-conv": lambda: bulk("Si", "diamond", 5.431, cubic=True),
    "Si-prim": lambda: bulk("Si", "diamond", 5.431),
    "MgO-conv": lambda: bulk("MgO", "rocksalt", 4.212, cubic=True),
    "MgO-prim": lambda: bulk("MgO", "rocksalt", 4.212),
    "NaCl-conv": lambda: bulk("NaCl", "rocksalt", 5.640, cubic=True),
}


def _sg(atoms, symprec=1e-2):
    import spglib
    return spglib.get_symmetry_dataset((atoms.cell.array, atoms.get_scaled_positions(),
                                        atoms.numbers), symprec=symprec).number


def _bond_lengths(atoms, bonds):
    c = atoms.cell.array
    return np.array([np.linalg.norm(atoms.positions[j] + np.asarray(s) @ c - atoms.positions[i])
                     for i, j, s in bonds])


@pytest.mark.parametrize("name, bond_only", [("Si-conv", 6), ("Si-prim", 0),
                                             ("MgO-conv", 9), ("MgO-prim", 0),
                                             ("NaCl-conv", 9)])
def test_repulsion_removes_the_floppy_modes_of_bond_springs(name, bond_only):
    atoms = CELLS[name]()
    assert C.floppy_modes(atoms, with_repulsion=False) == bond_only
    assert C.floppy_modes(atoms) == 0


@pytest.mark.parametrize("held", ["shown", "lenient"])
@pytest.mark.parametrize("name", list(CELLS))
def test_rattled_cell_moves_back_toward_its_bond_lengths(name, held):
    """As shown, the springs hold the bonds the viewer draws (1.1.6): this rattle stretches one bond past
    the default 1.15, so the document's rule is set as Build › Calculate Bonds… would set it, to draw them
    all. Lenient (Clean › Bonds held, 1.2 x) holds them all at the default criteria."""
    ideal = CELLS[name]()
    bonds = C.bonded_pairs(ideal)
    ref = _bond_lengths(ideal, bonds)
    rattled = ideal.copy()
    rattled.rattle(0.08, seed=3)
    err0 = np.sqrt(((_bond_lengths(rattled, bonds) - ref) ** 2).mean())
    rule = S.bond_rule({"bondTolerance": 1.2}) if held == "shown" else None
    assert set(C.held_bonds(rattled, held, rule)) == set(bonds)
    res = C.clean(rattled, keep_symmetry=False, fmax=0.005, max_steps=3000, bond_rule=rule, bonds_held=held)
    err1 = np.sqrt(((_bond_lengths(res["atoms"], bonds) - ref) ** 2).mean())
    assert res["converged"] and err1 < 0.3 * err0 and res["bonds_held"] == held, (err0, err1)
    if name.startswith(("Si", "MgO")):    # rigid and stable: the group comes back
        assert _sg(res["atoms"]) == _sg(ideal)


def test_bonds_held_modes():
    """Clean › Bonds held: As shown is the viewer's rule (a stretched bond the view no longer draws is not
    held), Lenient 1.2 x the bond radii with the default criteria, None no bond (the repulsion alone)."""
    ideal = CELLS["Si-conv"]()
    rattled = ideal.copy()
    rattled.rattle(0.08, seed=3)
    shown, lenient = set(C.held_bonds(rattled)), set(C.held_bonds(rattled, "lenient"))
    assert shown < lenient and len(lenient) == len(C.bonded_pairs(ideal)) and lenient - shown
    assert C.held_bonds(rattled, "none") == []
    # Lenient ignores the document's criteria; As shown takes them
    never = S.bond_rule({"bondPairs": {"Si-Si": {"never": True}}})
    assert C.held_bonds(rattled, "shown", never) == [] and set(C.held_bonds(rattled, "lenient", never)) == lenient
    # metals: Lenient bonds a structure of metals only, as the viewer does, and still no cation pair of a compound
    assert len(C.held_bonds(bulk("Cu", "fcc", 3.61, cubic=True), "lenient")) == 24
    mgo = CELLS["MgO-conv"]()
    sym = mgo.get_chemical_symbols()
    assert {sym[i] + sym[j] for i, j, _ in C.held_bonds(mgo, "lenient")} <= {"MgO", "OMg"}
    for bad in ("all", "Shown", 1):
        with pytest.raises(ValueError, match="bonds held is one of shown, lenient, none"):
            C.check_bonds_held(bad)
    assert C.check_bonds_held(None) == C.check_bonds_held("") == "shown"


def test_bonds_held_none_runs_the_repulsion_alone():
    """None: no springs, close pairs only push apart; a lone atom or a rule that bonds nothing runs too
    (nothing to clean is for As shown and Lenient)."""
    from ase import Atoms
    pair = Atoms("Si2", positions=[[0, 0, 0], [1.2, 0, 0]], cell=[12, 12, 12], pbc=True)    # far too close
    res = C.clean(pair, keep_symmetry=False, bonds_held="none", max_steps=3000)
    assert res["n_bonds"] == 0 and res["bonds_held"] == "none"
    assert res["atoms"].get_distance(0, 1, mic=True) > 1.2 + 0.5
    springs = C.clean(pair, keep_symmetry=False, max_steps=3000)        # As shown: Si-Si 1.2 Å is a bond
    assert springs["n_bonds"] == 1 and springs["atoms"].get_distance(0, 1, mic=True) < res["atoms"].get_distance(0, 1, mic=True)
    lone = Atoms("Ar", cell=[10, 10, 10], pbc=True)
    assert C.clean(lone, bonds_held="none")["converged"]
    for held in ("shown", "lenient"):
        with pytest.raises(C.NothingToClean):
            C.clean(lone, bonds_held=held)


def test_symmetric_expansion_keeps_the_group_and_shrinks_back():
    ideal = CELLS["MgO-conv"]()
    big = ideal.copy()
    big.set_cell(ideal.cell.array * 1.08, scale_atoms=True)
    res = C.clean(big, keep_symmetry=True, relax_cell=True, symprec=1e-5, max_steps=2000)
    out = res["atoms"]
    assert _sg(out, 1e-5) == 225
    a = out.cell.cellpar()[0]
    assert a < big.cell.cellpar()[0] - 0.1          # moves toward 2 x (R_Mg + R_O) = 4.14 A
    assert a == pytest.approx(2 * (1.41 + 0.66), abs=0.05)


def test_breathing_distortion_keeps_fd3m_with_fixsymmetry():
    si = CELLS["Si-prim"]()
    si.set_cell(si.cell.array * 1.05, scale_atoms=True)
    res = C.clean(si, keep_symmetry=True, relax_cell=True, symprec=1e-5, max_steps=2000)
    assert _sg(res["atoms"], 1e-5) == 227
    assert res["atoms"].get_volume() < si.get_volume()


@pytest.mark.parametrize("name", ["Si-conv", "MgO-conv", "NaCl-conv"])
def test_ideal_cell_with_default_options_does_not_move(name):
    ideal = CELLS[name]()
    res = C.clean(ideal)
    assert np.abs(res["atoms"].positions - ideal.positions).max() < 1e-3
    assert np.allclose(res["atoms"].cell.array, ideal.cell.array)   # relax_cell is off


def test_lone_atoms_and_unbonded_rules_have_nothing_to_clean():
    from ase import Atoms
    with pytest.raises(C.NothingToClean):
        C.clean(Atoms("Ar", cell=[10, 10, 10], pbc=True))
    with pytest.raises(C.NothingToClean):            # the document's rule bonds no pair
        C.clean(bulk("MgO", "rocksalt", 4.212), bond_rule=S.bond_rule({"bondPairs": {"Mg-O": {"never": True}}}))


def test_metals_clean_with_the_bonds_the_viewer_draws():
    """An all-metal cell (a metal, an alloy) has its metal-metal bonds drawn, so Clean holds them; a
    compound's cations stay apart unless Bonds between metals is on (geom/core.js metalMetalBonds)."""
    cu = bulk("Cu", "fcc", 3.61, cubic=True)
    alloy = bulk("Au", "fcc", 4.0, cubic=True)
    alloy.numbers[:2] = 29
    for metal in (cu, alloy):
        assert len(C.bonded_pairs(metal)) == 24
        r = metal.copy()
        r.rattle(0.03, seed=2)
        res = C.clean(r, keep_symmetry=False, fmax=0.005, max_steps=3000)
        assert res["converged"] and res["steps"] > 1
    mgo = bulk("MgO", "rocksalt", 4.212, cubic=True)
    assert {(n[i], n[j]) for n in [mgo.get_chemical_symbols()] for i, j, _ in C.bonded_pairs(mgo)} <= {("Mg", "O"), ("O", "Mg")}
    assert len(C.bonded_pairs(mgo, S.bond_rule({"bondMetalMetal": True}))) > len(C.bonded_pairs(mgo))


def test_time_limit_stops_the_run():
    big = CELLS["Si-conv"]().repeat((2, 2, 2))
    big.rattle(0.1, seed=1)
    res = C.clean(big, keep_symmetry=False, time_limit=0.0, fmax=1e-9)
    assert not res["converged"] and res["steps"] <= 2


def test_cell_filter_falls_back_on_old_ase(monkeypatch):
    import ase.filters
    monkeypatch.delattr(ase.filters, "FrechetCellFilter")
    f = C._cell_filter(CELLS["Si-prim"]())
    assert type(f).__name__ in ("ExpCellFilter", "UnitCellFilter")
