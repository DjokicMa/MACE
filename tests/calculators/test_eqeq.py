"""EQeq (mace/calculators/eqeq.py) against the authors' reference program on
IRMOF-1, plus the method's invariants on small systems."""
import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("scipy")
import calc_fixtures as CF  # noqa: E402
from calc_fixtures import preopt_sandbox  # noqa: E402,F401


def test_irmof1_matches_reference():
    from mace.calculators import eqeq
    a, ref = CF.irmof1()
    r = eqeq.charges(a)
    assert r["method"] == "ewald"
    q = r["charges"]
    assert np.abs(q - np.array(ref)).max() < 1e-4
    assert abs(q.sum()) < 1e-9
    zn = q[np.array(a.get_chemical_symbols()) == "Zn"]
    assert zn.min() == pytest.approx(1.2115, abs=1e-3)


def test_atomic_parameters():
    from mace.calculators import eqeq, eqeq_data as D
    p = eqeq.atomic_params(["H", "C", "Zn"])
    assert p["J"][0] == pytest.approx(13.598 + 2.0) and p["X"][0] == pytest.approx(0.5 * (13.598 - 2.0))
    ea, ip1 = D.IONIZATION["C"][:2]
    assert p["J"][1] == pytest.approx(ip1 - ea)
    ip2, ip3 = D.IONIZATION["Zn"][2:4]          # charge center 2
    assert p["J"][2] == pytest.approx(ip3 - ip2) and p["X"][2] == pytest.approx(0.5 * (ip3 + ip2) - 2 * (ip3 - ip2))
    p = eqeq.atomic_params(["Zn"], charge_centers={"Zn": 0})
    assert p["J"][0] == pytest.approx(D.IONIZATION["Zn"][1] - D.IONIZATION["Zn"][0])
    with pytest.raises(eqeq.EQeqError):
        eqeq.atomic_params(["Og"])


def test_molecule_symmetry_and_total_charge():
    from ase.build import molecule
    from mace.calculators import eqeq
    w = molecule("H2O")
    r = eqeq.charges(w)
    assert r["method"] == "nonperiodic"
    q = r["charges"]
    assert q[0] < 0 and q[1] == pytest.approx(q[2]) and abs(q.sum()) < 1e-10
    r = eqeq.charges(molecule("NH3"), total_charge=1.0)
    assert r["charges"].sum() == pytest.approx(1.0)


def test_slab_direct_and_ewald_refusal():
    from ase.build import fcc100
    from mace.calculators import eqeq
    s = fcc100("Cu", (2, 2, 2), vacuum=6.0)
    s.pbc = [True, True, False]
    r = eqeq.charges(s)
    assert r["method"] == "direct" and abs(r["charges"].sum()) < 1e-9
    with pytest.raises(eqeq.EQeqError):
        eqeq.charges(s, method="ewald")
