"""A1, the unit half: the spring model gives exactly what it gave before the
Clean tiers existed (positions, cell and step counts hashed on the code just before the Clean tiers), from
clean() and from the pool job, and the pool job now carries the charge."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("ase")
pytest.importorskip("spglib")

import preopt_fixtures as F  # noqa: E402
from preopt_fixtures import preopt_sandbox  # noqa: E402,F401  (autouse: temp root, XDG dirs)


@pytest.mark.parametrize("name", list(F.T0))
def test_clean_springs_matches_t0(name):
    from mace.gui import clean as C
    atoms = F.t0_structures()[name]
    if F.T0[name] is None:
        with pytest.raises(C.NothingToClean):
            C.clean(atoms, time_limit=60)
        return
    res = C.clean(atoms, time_limit=60)
    assert res["converged"]
    steps, digest = F.T0[name]
    assert res["steps"] == steps
    if F.t0_env_matches():
        assert F.spring_hash(res["atoms"], res["steps"]) == digest


@pytest.mark.parametrize("name", ["MgO_rattled", "FSI_box"])
def test_clean_job_springs_matches_t0(name):
    from mace.gui import builders as B
    atoms = F.t0_structures()[name]
    out = B.clean_job(cell=atoms.cell.array, frac=atoms.get_scaled_positions(wrap=False), numbers=atoms.numbers,
                      time_limit=60)
    rep = out["report"]
    assert rep["converged"] and rep["steps"] == F.T0[name][0]
    if F.t0_env_matches():
        from ase import Atoms
        a = Atoms(numbers=out["numbers"], cell=out["cell"], scaled_positions=out["frac"], pbc=True)
        ref = __import__("mace.gui.clean", fromlist=["clean"]).clean(atoms, time_limit=60)["atoms"]
        assert np.abs(a.positions - ref.positions).max() < 1e-9


def test_clean_job_records_the_charge():
    from mace.gui import builders as B
    atoms = F.fsi_box()
    out = B.clean_job(cell=atoms.cell.array, frac=atoms.get_scaled_positions(), numbers=atoms.numbers,
                      time_limit=60, charge=2, uhf=1)
    assert out["report"]["charge"] == 2 and out["report"]["uhf"] == 1
    plain = B.clean_job(cell=atoms.cell.array, frac=atoms.get_scaled_positions(), numbers=atoms.numbers, time_limit=60)
    assert plain["report"]["charge"] == 0
    assert np.asarray(out["frac"]).tobytes() == np.asarray(plain["frac"]).tobytes()     # springs ignore it


def test_adapter_factory_gets_options_and_is_closed():
    from mace.gui import clean as C

    class Stub(C.SpringModel):
        closed = 0

        def close(self):
            Stub.closed += 1
    seen = {}

    def factory(atoms, **opts):
        seen.update(opts)
        return Stub(C.bonded_pairs(atoms))
    C.ADAPTERS["stub-test"] = factory
    try:
        res = C.clean(F.rattled_mgo(), method="stub-test", calc_opts={"charge": -1}, time_limit=60)
        assert seen == {"charge": -1} and Stub.closed == 1 and res["method"] == "stub-test"

        def boom(atoms, **opts):
            s = Stub(C.bonded_pairs(atoms))
            s.calculate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("calc failed"))
            return s
        C.ADAPTERS["stub-test"] = boom
        with pytest.raises(RuntimeError):
            C.clean(F.rattled_mgo(), method="stub-test", time_limit=60)
        assert Stub.closed == 2                      # closed on the failure path too
    finally:
        C.ADAPTERS.pop("stub-test", None)
