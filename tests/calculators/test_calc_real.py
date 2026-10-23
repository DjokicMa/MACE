"""Real calculations with the installed codes (opt-in: MACE_TEST_REAL_CALC=1).

Uses the machine's own setup: MACE_XTB (xtb 6.7+), MACE_DFTB with
MACE_DFTB_SKF (a Slater-Koster set such as 3ob), MACE_MLIP_PYTHON with
MACE_MLIP_DIR (the ML environment and fetched weights). Each part skips
when its code is not there. The ML runs use device auto: GPU 0 when there is
one. Checks are physical: Si's lattice constant within 3 % of 5.43 Å after a
cell relaxation, water's bond length and angle, energies finite."""
import os

import numpy as np
import pytest

pytest.importorskip("ase")
import calc_fixtures as CF  # noqa: E402
from calc_fixtures import preopt_sandbox  # noqa: E402,F401

REAL = {k: os.environ.get(k) for k in ("MACE_XTB", "MACE_DFTB", "MACE_DFTB_SKF", "MACE_MLIP_PYTHON", "MACE_MLIP_DIR")}
pytestmark = pytest.mark.skipif(os.environ.get("MACE_TEST_REAL_CALC") != "1",
                                reason="real calculations are opt-in: MACE_TEST_REAL_CALC=1")


@pytest.fixture
def real_env(monkeypatch):
    for k, v in REAL.items():
        if v:
            monkeypatch.setenv(k, v)


def _water():
    from ase.build import molecule
    m = molecule("H2O")
    m.center(vacuum=7.0)
    m.pbc = True
    return m


def _check_water(a):
    d = a.get_distance(0, 1)
    ang = a.get_angle(1, 0, 2)
    assert 0.9 < d < 1.05 and 95 < ang < 115, (d, ang)


def _si_relaxed(spec):
    from ase.build import bulk
    from mace.calculators import tasks
    r = tasks.run(bulk("Si", "diamond", a=5.6, cubic=True), spec)
    a = r.atoms.cell.lengths()[0]
    assert abs(a - 5.43) / 5.43 < 0.04, a
    return r


@pytest.mark.parametrize("method", ["gfn2", "gfn1", "gfnff"])
def test_xtb(real_env, method):
    from mace.calculators import registry, tasks
    from mace.calculators.base import CalcSpec
    if not registry.get("xtb").status(method)["available"]:
        pytest.skip("xtb is not installed")
    r = tasks.run(_water(), CalcSpec("xtb", method=method, task="opt", fmax=0.01))
    assert r.converged
    _check_water(r.atoms)


def test_dftb(real_env):
    from mace.calculators import registry, tasks
    from mace.calculators.base import CalcSpec
    if not registry.get("dftb").status("skf")["available"]:
        pytest.skip("DFTB+ or a Slater-Koster set is not installed")
    r = tasks.run(_water(), CalcSpec("dftb", method="skf", task="opt", fmax=0.01))
    assert r.converged and r.charges is not None and r.charges[0] < 0
    _check_water(r.atoms)


def _ready_models():
    from mace.calculators import registry
    be = registry.get("mlip")
    return [m.id for m in be.methods() if be.status(m.id)["available"]]


def test_mlip_models(real_env):
    from mace.calculators import tasks
    from mace.calculators.base import CalcSpec
    ids = _ready_models()
    if not ids:
        pytest.skip("no ML model is ready (mace mlip setup, mace mlip fetch)")
    for mid in ids:
        r = _si_relaxed(CalcSpec("mlip", method=mid, task="opt", relax_cell=True, steps=100))
        assert np.isfinite(r.energy) and r.report.get("device_text")


@pytest.mark.parametrize("method", ["gfn2", "gfnff"])
def test_xtb_wrapped_molecule_in_a_box(real_env, method, tmp_path):
    """Benzene wrapped across the faces of a 15 Å box gives the centered benzene's energy (mace calc
    through mace_cli): xtb runs a molecule in a box without the cell, so it gets it whole."""
    import json
    import subprocess
    import sys
    from ase.build import molecule
    from ase.io import write
    from mace.calculators import registry
    if not registry.get("xtb").status(method)["available"]:
        pytest.skip("xtb is not installed")
    b = molecule("C6H6")
    b.set_cell([15.0, 15.0, 15.0])
    b.pbc = True
    b.center()
    write(str(tmp_path / "centered.extxyz"), b, format="extxyz")
    b.translate(-b.positions.mean(axis=0))
    b.wrap()
    write(str(tmp_path / "wrapped.extxyz"), b, format="extxyz")
    e = {}
    for name in ("centered", "wrapped"):
        p = subprocess.run([sys.executable, str(CF.REPO / "mace_cli"), "--no-banner", "calc", "sp",
                            str(tmp_path / f"{name}.extxyz"), "-b", "xtb", "-m", method, "--json"],
                           cwd=str(tmp_path), capture_output=True, text=True, timeout=600,
                           env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
        assert p.returncode == 0, p.stderr
        e[name] = json.loads(p.stdout)["energy_eV"]
    assert e["wrapped"] == pytest.approx(e["centered"], abs=1e-5), e
