"""Clean geometry: the spring pre-relaxation on distorted textbook cells."""
import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("spglib")

from ase.build import bulk  # noqa: E402

from mace.gui import clean as C  # noqa: E402

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


@pytest.mark.parametrize("name", list(CELLS))
def test_rattled_cell_moves_back_toward_its_bond_lengths(name):
    ideal = CELLS[name]()
    bonds = C.bonded_pairs(ideal)
    ref = _bond_lengths(ideal, bonds)
    rattled = ideal.copy()
    rattled.rattle(0.08, seed=3)
    err0 = np.sqrt(((_bond_lengths(rattled, bonds) - ref) ** 2).mean())
    res = C.clean(rattled, keep_symmetry=False, fmax=0.005, max_steps=3000)
    err1 = np.sqrt(((_bond_lengths(res["atoms"], bonds) - ref) ** 2).mean())
    assert res["converged"] and err1 < 0.3 * err0, (err0, err1)
    if name.startswith(("Si", "MgO")):    # rigid and stable: the group comes back
        assert _sg(res["atoms"]) == _sg(ideal)


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


def test_metals_and_lone_atoms_have_nothing_to_clean():
    with pytest.raises(C.NothingToClean):
        C.clean(bulk("Cu", "fcc", 3.61))
    from ase import Atoms
    with pytest.raises(C.NothingToClean):
        C.clean(Atoms("Ar", cell=[10, 10, 10], pbc=True))


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
