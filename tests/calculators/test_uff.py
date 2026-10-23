"""Native UFF / UFF4MOF (mace/calculators/uff.py, uff_typing.py).

* energies and gradients equal RDKit's UFF on 30 molecules (stored RDKit
  results, so RDKit is not needed; with RDKit installed the comparison is
  also made live);
* forces and the stress tensor equal finite differences of the energy, for
  a molecule, a triclinic cell and IRMOF-1;
* the UFF4MOF types of IRMOF-1 (Zn4O, carboxylates) and the paddlewheel;
* physical checks: the C_3-C_3 rest length, the ethane torsion barrier."""
import numpy as np
import pytest

pytest.importorskip("ase")
import calc_fixtures as CF  # noqa: E402
from calc_fixtures import preopt_sandbox  # noqa: E402,F401

KCAL = 0.0433641043     # eV per kcal/mol (ase.units)


def _uff(atoms, mode="uff", **kw):
    from mace.calculators.uff import UFFCalculator
    a = atoms.copy()
    a.calc = UFFCalculator(mode=mode, **kw)
    return a


@pytest.mark.parametrize("mol", CF.rdkit_reference(), ids=lambda m: m["smiles"])
def test_matches_rdkit(mol):
    from ase import Atoms
    a = _uff(Atoms(mol["symbols"], positions=mol["positions"]))
    e = a.get_potential_energy() / KCAL
    assert e == pytest.approx(mol["energy"], abs=1e-4)
    g = -a.get_forces() / KCAL
    assert np.abs(g.ravel() - np.array(mol["gradient"])).max() < 1e-3


def test_matches_rdkit_live():
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem
    from ase import Atoms
    for smi in ("CC(=O)Nc1ccc(O)cc1", "C=CC(=O)OCC", "c1ccc(cc1)C#N"):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=3)
        ff = AllChem.UFFGetMoleculeForceField(m)
        a = _uff(Atoms([x.GetSymbol() for x in m.GetAtoms()], positions=m.GetConformer().GetPositions()))
        assert a.get_potential_energy() / KCAL == pytest.approx(ff.CalcEnergy(), abs=1e-4)


def _fd_forces(a, idx, h=1e-5):
    err = 0.0
    f = a.get_forces()
    for k in idx:
        for x in range(3):
            b = a.copy()
            b.calc = a.calc
            b.positions[k, x] += h
            ep = b.get_potential_energy()
            b.positions[k, x] -= 2 * h
            em = b.get_potential_energy()
            err = max(err, abs(-(ep - em) / (2 * h) - f[k, x]))
    return err


def _fd_stress(a, h=1e-5):
    s = a.get_stress(voigt=False)
    c0, V = a.cell.array.copy(), a.get_volume()
    err = 0.0
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        eps = np.zeros((3, 3))
        eps[i, j] = eps[j, i] = h
        vals = []
        for sg in (1, -1):
            b = a.copy()
            b.calc = a.calc
            b.set_cell(c0 @ (np.eye(3) + sg * eps), scale_atoms=True)
            vals.append(b.get_potential_energy())
        num = (vals[0] - vals[1]) / (2 * h) / V / (2 if i != j else 1)
        err = max(err, abs(num - s[i, j]))
    return err


def test_forces_and_stress_triclinic():
    from ase.build import molecule
    from ase import Atoms
    m = molecule("CH3COOH")
    a = Atoms(m.get_chemical_symbols(), positions=m.positions + 3.0,
              cell=[[9.0, 0, 0], [1.5, 8.5, 0], [0.7, -1.1, 9.5]], pbc=True)
    a.rattle(0.05, seed=2)
    a = _uff(a, cutoff=8.0, charges="eqeq")
    assert _fd_forces(a, range(len(a))) < 1e-5
    assert _fd_stress(a) < 1e-6
    assert set(a.calc.parts) >= {"bond", "angle", "torsion", "inversion", "vdw", "coulomb"}


def test_irmof1_types_forces_stress():
    a, _ = CF.irmof1()
    a = _uff(a, mode="uff4mof")
    a.get_potential_energy()
    t = a.calc.topology
    assert t.type_counts() == {"C_R": 192, "H_": 96, "O_2": 96, "O_3_f": 8, "Zn3f2": 32}
    assert not t.warnings
    assert _fd_forces(a, [0, 1, 50, 200, 400]) < 1e-5
    assert _fd_stress(a) < 1e-5
    # Zn-O bonds of UFF4MOF have order 0.5
    z = a.numbers
    assert {o for i, j, s, o in t.bonds if 30 in (z[i], z[j])} == {0.5}
    # plain UFF: the generic tetrahedral zinc type
    b = _uff(CF.irmof1()[0], mode="uff")
    b.get_potential_energy()
    assert b.calc.topology.type_counts()["Zn3+2"] == 32


def test_paddlewheel_types():
    """A Cu2(O2CH)4 paddlewheel in a box: Cu 4+2 (square planar plus Cu-Cu)."""
    from ase import Atoms
    cu = 1.3
    pos, sym = [(0, 0, cu), (0, 0, -cu)], ["Cu", "Cu"]
    for k in range(4):
        ang = k * np.pi / 2
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
        for z0, s in ((cu + 0.1, 1), (-cu - 0.1, -1)):
            pos.append(tuple(d * 1.95 + [0, 0, z0 - s * 0.1]))
            sym.append("O")
        pos.append(tuple(d * 2.55))
        sym.append("C")
        pos.append(tuple(d * 3.6))
        sym.append("H")
    a = Atoms(sym, positions=np.array(pos) + 8.0, cell=[16.0] * 3, pbc=True)
    a = _uff(a, mode="uff4mof")
    a.get_potential_energy()
    tc = a.calc.topology.type_counts()
    assert tc["Cu4+2"] == 2 and tc["C_R"] == 4 and tc["O_2"] == 8


def test_rest_length_and_ethane_barrier():
    from mace.calculators import uff_params as P
    from mace.calculators.uff import rest_length
    assert rest_length("C_3", "C_3", 1.0, P.UFF_DATA) == pytest.approx(1.514)
    from ase.build import molecule
    e = []
    for phi in (0.0, 60.0):                     # one methyl turned by 60 degrees: staggered <-> eclipsed
        m = molecule("C2H6")
        m.set_dihedral(2, 0, 1, 5, m.get_dihedral(2, 0, 1, 5) + phi, indices=[1, 5, 6, 7])
        a = _uff(m)
        a.get_potential_energy()
        e.append(a.calc.parts["torsion"] / KCAL)
    # nine H-C-C-H torsions, V = 2.119 kcal/mol each divided by nine: the barrier is V
    assert abs(e[1] - e[0]) == pytest.approx(2.119, abs=2e-3) and min(e) == pytest.approx(0.0, abs=1e-3)


def test_backend_runs_tasks(tmp_path):
    from ase.build import molecule
    from mace.calculators import tasks
    from mace.calculators.base import CalcSpec
    m = molecule("CH3OH")
    m.center(vacuum=5.0)
    m.pbc = True
    r = tasks.run(m, CalcSpec("uff", task="opt", fmax=0.01, steps=300))
    assert r.converged and r.fmax < 0.01 and r.report["atom_types"]["O_3"] == 1
    f = tasks.run(r.atoms, CalcSpec("uff", task="freq"))
    nu = np.sort(f.frequencies)
    assert np.abs(nu[:6]).max() < 30.0 and nu[-1] > 2500.0       # 6 near-zero modes; O-H/C-H stretches
    assert f.thermo["zpe_eV"] > 0.5


def _graphene_cell(a=2.46, box=20.0, pbc=True):
    from ase import Atoms
    return Atoms("C2", cell=[[a, 0, 0], [-a / 2, a * 3 ** 0.5 / 2, 0], [0, 0, box]],
                 scaled_positions=[[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]], pbc=pbc)


@pytest.mark.parametrize("element", ["C", "BN"])
def test_rings_through_periodic_images(element):
    """The 6-rings of graphene's (h-BN's) two-atom cell pass through the same atoms in other images:
    they are found, so every bond is aromatic (1.5), the in-plane stress is isotropic and the energy
    per atom is the 4x4 supercell's."""
    from mace.calculators import uff_typing
    g = _graphene_cell(a=2.46 if element == "C" else 2.50)
    if element == "BN":
        g.set_chemical_symbols(["B", "N"])
    top = uff_typing.build(g, "uff")
    assert len(top.rings) == 1 and top.aromatic == [True], (top.rings, top.aromatic)
    assert len(top.bonds) == 3 and {o for *_, o in top.bonds} == {1.5}, top.bonds
    big = g.repeat((4, 4, 1))
    assert len(uff_typing.build(big, "uff").rings) == 16
    a, b = _uff(g), _uff(big)
    s = a.get_stress()
    assert s[0] == pytest.approx(s[1], rel=1e-6, abs=1e-8) and abs(s[5]) < 1e-8, s
    assert a.get_potential_energy() / 2 == pytest.approx(b.get_potential_energy() / 32, abs=1e-6)
    assert np.allclose(b.get_stress(), s, atol=1e-8)


def test_graphene_cell_relaxation_keeps_the_hexagonal_cell(tmp_path):
    """mace calc opt --relax-cell on the two-atom cell of graphene (pbc "T T F") keeps gamma at 120 deg."""
    import os
    import subprocess
    import sys
    from ase.io import read, write
    write(str(tmp_path / "g.extxyz"), _graphene_cell(a=2.6, box=16.0, pbc=[True, True, False]), format="extxyz")
    p = subprocess.run([sys.executable, str(CF.REPO / "mace_cli"), "--no-banner", "calc", "opt",
                        str(tmp_path / "g.extxyz"), "-b", "uff", "--relax-cell"], cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=300,
                       env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert p.returncode == 0, p.stdout + p.stderr
    c = read(str(tmp_path / "g_uff-uff4mof_opt.extxyz")).cell.cellpar()
    assert c[0] == pytest.approx(c[1], abs=1e-3) and c[5] == pytest.approx(120.0, abs=0.05), c
