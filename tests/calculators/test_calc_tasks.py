"""The task drivers (mace/calculators/tasks.py): stopping early keeps the
partial result, the write-check hook, the report is JSON, input reading
refuses a deck that edits its geometry, and periodicity sees vacuum."""
import json

import numpy as np
import pytest

pytest.importorskip("ase")
import calc_fixtures as CF  # noqa: E402
from calc_fixtures import preopt_sandbox  # noqa: E402,F401


def _rattled():
    from ase.build import bulk
    a = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 1, 1))
    a.rattle(0.1, seed=4)
    return a


def test_stop_early_keeps_partial():
    from mace.calculators import tasks
    from mace.calculators.base import CalcSpec
    calls = {"n": 0}

    def canceled():
        calls["n"] += 1
        return calls["n"] > 3
    r = tasks.run(_rattled(), CalcSpec("uff", task="opt", fmax=1e-6, steps=500), canceled=canceled)
    assert r.stopped == "canceled" and not r.converged and r.steps < 10 and r.atoms is not None
    r = tasks.run(_rattled(), CalcSpec("uff", task="md", md_steps=10 ** 6, time_limit=0.5))
    assert r.stopped == "time limit" and r.md["steps_done"] < 10 ** 6


def test_outputs_and_check_hook(tmp_path):
    from mace.calculators import tasks
    from mace.calculators.base import CalcSpec
    spec = CalcSpec("eqeq", task="charges")
    r = tasks.run(_rattled(), spec)
    seen = []
    files = tasks.write_outputs(r, tmp_path, "nacl", spec, check=seen.append)
    assert {p.name for p in seen} == set(files.values())
    rep = json.loads((tmp_path / files["report"]).read_text())
    assert rep["charges"] and rep["provenance"]["citation"][0]["doi"] == "10.1021/jz3008485"
    text = (tmp_path / files["charges_cif"]).read_text()
    assert "_atom_site_charge" in text and text.count("Na") >= 4

    def refuse(p):
        raise PermissionError("no")
    with pytest.raises(PermissionError):
        tasks.write_outputs(r, tmp_path / "x", "nacl", spec, check=refuse)
    assert not (tmp_path / "x").exists()


def test_periodicity():
    from ase.build import bulk, fcc111, molecule
    from mace.calculators.tasks import periodicity
    assert periodicity(bulk("Si")) == 3
    s = fcc111("Cu", (2, 2, 3), vacuum=8.0)
    s.pbc = True
    assert periodicity(s) == 2
    m = molecule("C6H6")
    assert periodicity(m) == 0
    m.center(vacuum=6.0)
    m.pbc = True
    assert periodicity(m) == 0


def test_structio_refuses_geometry_edits(tmp_path):
    import test_preopt_cli as T
    from mace.calculators import structio
    from mace.calculators.base import CalcError
    p = tmp_path / "defect.d12"
    p.write_text(T.MGO_DECK.format(edits="SUPERCEL\n2 0 0\n0 2 0\n0 0 2\n", basis_extra="", scf_extra=""))
    with pytest.raises(CalcError) as e:
        structio.read(p)
    assert "SUPERCEL" in str(e.value)
    q = tmp_path / "plain.d12"
    q.write_text(T.MGO_DECK.format(edits="", basis_extra="", scf_extra=""))
    atoms, sg, kinds = structio.read(q)
    assert len(atoms) == 8 and sg is not None


def test_registry_resolve_and_spec():
    from mace.calculators import registry
    assert registry.resolve("uff4mof") == ("uff", "uff4mof")
    assert registry.resolve("gfn2") == ("xtb", "gfn2")
    assert registry.resolve("mlip:orb-v3-conservative-inf-mpa") == ("mlip", "orb-v3-conservative-inf-mpa")
    assert registry.resolve("xtb", "gfn1") == ("xtb", "gfn1")
    rows = [b.describe() for b in registry.all_backends()]
    assert all(r["methods"] for r in rows)
    json.dumps(rows, default=str)
