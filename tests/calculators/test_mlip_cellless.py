"""What the ML worker is sent (cell and pbc) for a structure with no cell or
with pbc on some axes only: a plain XYZ, an extended XYZ with pbc flags, and
CRYSTAL MOLECULE / POLYMER / SLAB outputs, through `mace calc` and through
`mace preopt`'s ML tier (the GUI's Calculators and Clean run the same code).

A plain XYZ used to reach MACE-MP (mace-torch) as pbc=True with an all-zero
cell: ``lowdim.boxed`` could not stretch a zero vector into a vacuum box, and
mace-torch's neighbor list takes a zero cell for a 1 Å cube, so the molecule
saw its own images 1 Å away ("bigforce: 1625 eV/Å"). A molecule now reaches
mace-torch, ORB and the charge/spin models as pbc=False; CHGNet and MatterSim,
which always build a 3D crystal, get a real box with lowdim.VACUUM Å around it.
"""
import json
import lzma
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("ase")
import calc_fixtures as CF  # noqa: E402
from calc_fixtures import preopt_sandbox  # noqa: E402,F401

REPO = CF.REPO
CLI = REPO / "mace_cli"
WATER = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n"
WATER_BOXED = ('3\nLattice="15 0 0 0 15 0 0 0 15" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
               "O 7.5 7.5 7.6173\nH 7.5 8.2572 7.0308\nH 7.5 6.7428 7.0308\n")
HBN = ('2\nLattice="2.5 0 0 -1.25 2.165 0 0 0 6" Properties=species:S:1:pos:R:3 pbc="T T F"\n'
       "B 0 0 3\nN 1.25 0.7217 3\n")
OUTS = {"EC_opt.out": (REPO / "examples" / "suite" / "5-molecule-0d" / "EC" / "OPT" / "EC_opt.out.xz", 0),
        "polyacetylene_opt.out": (REPO / "examples" / "suite" / "4-polymer-1d" / "polyacetylene" / "OPT"
                                  / "polyacetylene_opt.out.xz", 1),
        "graphene_lg37.out": (REPO / "tests" / "data" / "low_dim_groups" / "graphene_lg37.out", 2)}


def mace(*args, cwd=None, env=None, timeout=300):
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_XTB", "MACE_DFTB", "MACE_MLIP_PYTHON")}
    e.update(env or {})
    p = subprocess.run([sys.executable, str(CLI), "--no-banner", *args], cwd=cwd, env=e, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def _sent(log):
    return [json.loads(x) for x in log.read_text().splitlines()]


def _write_out(tmp_path, name):
    src, dim = OUTS[name]
    dst = tmp_path / name
    dst.write_bytes(lzma.decompress(src.read_bytes()) if src.suffix == ".xz" else src.read_bytes())
    return dst, dim


def _assert_boxed(msg, atoms, distances=False):
    """pbc on every axis, the periodic vectors kept, VACUUM Å or more along the open axes (and, for the
    first call on a molecule, the interatomic distances unchanged, with no image nearer than its own atoms)."""
    from ase import Atoms
    from mace.preopt import lowdim
    cell = np.array(msg["cell"])
    assert msg["pbc"] == [True] * 3
    assert abs(np.linalg.det(cell)) > 1.0
    for k in range(3):
        if atoms.pbc[k]:
            assert np.allclose(cell[k], atoms.cell.array[k])
        else:
            assert np.linalg.norm(cell[k]) >= lowdim.VACUUM - 1e-6
    b = Atoms(atoms.numbers, positions=msg["positions"], cell=cell, pbc=True)
    a = atoms.copy()
    if distances and not any(a.pbc):
        assert np.allclose(b.get_all_distances(), a.get_all_distances(), atol=1e-8)
        assert np.allclose(b.get_all_distances(mic=True), a.get_all_distances(), atol=1e-8)


@pytest.fixture
def fake_ml(tmp_path, monkeypatch):
    """The fake ML worker serves every registry model (weights taken as present), logging what it is sent."""
    from mace.preopt import detect
    py = CF.F.make_fake_python(tmp_path / "venv")
    weights = tmp_path / "weights.model"
    weights.write_bytes(b"stand-in weights")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("MACE_MLIP_PYTHON", str(py))
    monkeypatch.setenv("FAKE_CALC_LOG", str(log))
    monkeypatch.setattr(detect, "weights_status", lambda m: {"status": "ready", "path": str(weights)})
    monkeypatch.setattr(detect, "mlip_status", lambda m, recheck=False, **k: {
        "status": "ready", "python": str(py), "versions": {}, "weights_path": str(weights)})
    return log


def _calc_sp(atoms, model):
    from mace.calculators import tasks
    from mace.calculators.base import CalcSpec
    return tasks.run(atoms, CalcSpec("mlip", method=model, task="sp", device="cpu", threads=1))


def _preopt(atoms, model):
    """mace preopt's ML tier (and the GUI's Clean), for the models it offers (ORB-v3, MatterSim: mace calc only)."""
    from mace.preopt import core, registry
    if registry.get(model) is None:
        return None
    r = core.preoptimize(atoms, f"mlip:{model}", steps=3)
    tried = [x for x in r.attempts if x["method"] == f"mlip:{model}"][0]
    assert tried["status"] in ("ok", "rejected"), tried
    return r


def test_boxed_gives_a_cell_less_molecule_a_real_box():
    """The root cause: a zero cell came back from lowdim.boxed as a zero cell with pbc on."""
    from ase import Atoms
    from mace.preopt import lowdim
    w = Atoms("OH2", positions=[[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]])
    assert not w.cell.any() and not any(w.pbc)
    b = lowdim.boxed(w)
    assert all(b.pbc) and b.cell.handedness == 1
    assert min(b.cell.lengths()) >= lowdim.VACUUM - 1e-6
    assert np.allclose(b.get_all_distances(mic=True), w.get_all_distances())
    # a slab without c, a polymer without b and c: normal directions, right-handed
    s = Atoms("BN", positions=[[0, 0, 0], [1.25, 0.72, 0]], cell=[[2.5, 0, 0], [-1.25, 2.165, 0], [0, 0, 0]],
              pbc=[True, True, False])
    b = lowdim.boxed(s)
    assert np.allclose(b.cell.array[:2], s.cell.array[:2]) and np.allclose(b.cell.array[2], [0, 0, lowdim.VACUUM])
    p = Atoms("C2", positions=[[0, 0, 0], [1, 1, 1]], cell=[[1.5, 1.5, 1.5], [0, 0, 0], [0, 0, 0]],
              pbc=[True, False, False])
    b = lowdim.boxed(p)
    c = b.cell.array
    assert np.allclose(c[0], p.cell.array[0]) and b.cell.handedness == 1
    assert abs(c[0] @ c[1]) < 1e-9 and abs(c[0] @ c[2]) < 1e-9 and abs(c[1] @ c[2]) < 1e-9
    assert min(b.cell.lengths()[1:]) >= lowdim.VACUUM - 1e-6


def test_cell_less_xyz_reaches_mace_torch_as_a_molecule(tmp_path):
    """`mace calc sp water.xyz` with a mace-torch model: the worker gets pbc off and the zero cell, and the
    energy equals that of the same molecule in a 15 Å box."""
    (tmp_path / "water.xyz").write_text(WATER)
    (tmp_path / "boxed.extxyz").write_text(WATER_BOXED)
    model = tmp_path / "mine.model"
    model.write_bytes(b"a model file the user supplies")
    py = CF.F.make_fake_python(tmp_path / "venv")
    energies = {}
    for name in ("water.xyz", "boxed.extxyz"):
        log = tmp_path / f"{name}.jsonl"
        rc, out, err = mace("calc", "sp", name, "-b", "mlip", "--model-file", str(model), "--model-backend",
                            "mace-torch", "--model-license-ok", "--device", "cpu", "--json", "-o", "out",
                            cwd=tmp_path, env={"MACE_MLIP_PYTHON": str(py), "FAKE_CALC_LOG": str(log)})
        assert rc == 0, err
        energies[name] = json.loads(out)["energy_eV"]
        sent = _sent(log)
        assert sent
        if name == "water.xyz":
            assert all(m["pbc"] == [False] * 3 and not np.any(m["cell"]) for m in sent), sent
        else:
            assert all(m["pbc"] == [True] * 3 for m in sent)
    assert energies["water.xyz"] == pytest.approx(energies["boxed.extxyz"], abs=1e-9)


@pytest.mark.parametrize("model", ["acesuit-mace-mpa-0-medium", "orb-v3-conservative-inf-mpa"])
def test_molecule_reaches_open_boundary_models_without_a_cell(fake_ml, tmp_path, model):
    from mace.calculators import structio
    (tmp_path / "water.xyz").write_text(WATER)
    atoms = structio.read(tmp_path / "water.xyz")[0]
    _calc_sp(atoms, model)
    _preopt(atoms, model)
    sent = _sent(fake_ml)
    assert sent and all(m["pbc"] == [False] * 3 and not np.any(m["cell"]) for m in sent), sent


@pytest.mark.parametrize("model", ["chgnet-0.3.0", "mattersim-v1.0.0-1m"])
def test_molecule_for_crystal_only_models_gets_a_vacuum_box(fake_ml, tmp_path, model):
    from mace.calculators import structio
    (tmp_path / "water.xyz").write_text(WATER)
    atoms = structio.read(tmp_path / "water.xyz")[0]
    _calc_sp(atoms, model)
    _preopt(atoms, model)
    sent = _sent(fake_ml)
    assert sent
    _assert_boxed(sent[0], atoms, distances=True)
    for m in sent:
        _assert_boxed(m, atoms)


def test_extxyz_slab_with_pbc_flags_is_boxed_along_c(fake_ml, tmp_path):
    from mace.calculators import structio
    (tmp_path / "hbn.extxyz").write_text(HBN)
    atoms = structio.read(tmp_path / "hbn.extxyz")[0]
    assert list(atoms.pbc) == [True, True, False]
    _calc_sp(atoms, "acesuit-mace-mpa-0-medium")
    _preopt(atoms, "acesuit-mace-mpa-0-medium")
    for m in _sent(fake_ml):
        _assert_boxed(m, atoms)


@pytest.mark.parametrize("name", sorted(OUTS))
def test_crystal_low_dim_outs(fake_ml, tmp_path, name):
    """A MOLECULE output reaches mace-torch with pbc off (its drawing box unused) and CHGNet boxed; a POLYMER
    or SLAB output reaches both boxed along its open axes, its periodic vectors kept."""
    from mace.calculators import structio
    path, dim = _write_out(tmp_path, name)
    atoms = structio.read(path)[0]
    assert sum(bool(x) for x in atoms.pbc) == dim
    for model in ("acesuit-mace-mpa-0-medium", "chgnet-0.3.0"):
        fake_ml.write_text("")
        _calc_sp(atoms, model)
        _preopt(atoms, model)
        sent = _sent(fake_ml)
        assert sent
        for m in sent:
            if dim == 0 and model.startswith("acesuit"):
                assert m["pbc"] == [False] * 3
            else:
                _assert_boxed(m, atoms)
        if dim == 0 and model.startswith("chgnet"):
            _assert_boxed(sent[0], atoms, distances=True)
