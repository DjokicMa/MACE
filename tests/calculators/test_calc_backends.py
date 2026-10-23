"""The backends' plumbing without the real programs: the xtb command lines
(a stand-in xtb), the DFTB+ input (a stand-in dftb+, the 3ob README rules),
the ML potentials through the fake ML worker of the preopt tests (device
auto gives the worker GPU 0 only), and the registry's license rules."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")
import calc_fixtures as CF  # noqa: E402
from calc_fixtures import preopt_sandbox  # noqa: E402,F401

FAKE_XTB = r'''#!{exe}
import json, os, sys
args = sys.argv[1:]
if "--version" in args:
    print("      * xtb version 6.7.1 (fake) compiled by x on 2024")
    sys.exit(0)
log = os.environ.get("FAKE_XTB_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(json.dumps(args) + "\n")
keep = os.environ.get("FAKE_XTB_INPUTS")
if keep:
    with open(keep, "a") as fh:
        fh.write(json.dumps(open(args[0]).read()) + "\n")
fn = args[0]
lines = open(fn).read().splitlines()
if fn.endswith(".xyz"):
    n = int(lines[0]); periodic = False
else:
    n = sum(int(x) for x in lines[6].split()); periodic = True
print("          | TOTAL ENERGY             -12.500000000000 Eh   |")
with open("gradient", "w") as fh:
    fh.write("$grad\n  cycle =      1    SCF energy =   -12.5   |dE/xyz| =  0.0\n")
    for _ in range(n):
        fh.write("  0.0 0.0 0.0 x\n")
    for _ in range(n):
        fh.write("  0.0 0.0 0.0\n")
    fh.write("$end\n")
if periodic:
    with open("gradlatt", "w") as fh:
        fh.write("$gradlatt\n  cycle =      1\n")
        for _ in range(6):
            fh.write("  0.0 0.0 0.0\n")
        fh.write("$end\n")
if "--gfnff" not in args:
    with open("charges", "w") as fh:
        fh.write("\n".join(["0.1", "-0.1"] * (n // 2) + (["0.0"] if n % 2 else [])) + "\n")
'''


@pytest.fixture
def fake_xtb(tmp_path, monkeypatch):
    p = tmp_path / "bin" / "xtb"
    p.parent.mkdir()
    p.write_text(FAKE_XTB.format(exe=sys.executable))
    p.chmod(0o755)
    monkeypatch.setenv("MACE_XTB", str(p))
    log = tmp_path / "xtb.log"
    monkeypatch.setenv("FAKE_XTB_LOG", str(log))
    return log


def _water_box():
    from ase.build import molecule
    m = molecule("H2O")
    m.center(vacuum=7.0)
    m.pbc = True
    return m


def test_xtb_command_lines(fake_xtb):
    from ase.build import bulk
    from mace.calculators import tasks
    from mace.calculators.base import CalcSpec
    r = tasks.run(_water_box(), CalcSpec("xtb", method="gfn2", task="sp", charge=-1, uhf=1))
    assert r.energy == pytest.approx(-12.5 * 27.211386, rel=1e-6)
    assert r.charges is not None and len(r.charges) == 3
    tasks.run(bulk("Si", "diamond", a=5.43), CalcSpec("xtb", method="gfn1", task="sp"))
    tasks.run(_water_box(), CalcSpec("xtb", method="gfnff", task="sp"))
    calls = [json.loads(x) for x in fake_xtb.read_text().splitlines()]
    assert calls[0][0] == "mol.xyz" and calls[0][1:3] == ["--gfn", "2"] and "--tblite" not in calls[0]
    assert calls[0][calls[0].index("--chrg") + 1] == "-1" and calls[0][calls[0].index("--uhf") + 1] == "1"
    assert calls[1][0] == "POSCAR" and calls[1][1:4] == ["--gfn", "1", "--tblite"]
    assert calls[2][0] == "mol.xyz" and "--gfnff" in calls[2]


def test_xtb_molecule_in_a_box_is_made_whole(fake_xtb, tmp_path, monkeypatch):
    """A molecule in a 3D box runs as a molecule (no cell): xtb gets it whole, not the wrapped
    coordinates split across the box's faces."""
    from ase.build import molecule
    from mace.calculators import tasks
    from mace.calculators.base import CalcSpec
    inputs = tmp_path / "inputs.log"
    monkeypatch.setenv("FAKE_XTB_INPUTS", str(inputs))
    b = molecule("C6H6")
    b.set_cell([15.0, 15.0, 15.0])
    b.pbc = True
    b.translate(-b.positions.mean(axis=0))          # centered on a corner: split over all eight cells
    b.wrap()
    assert np.ptp(b.positions, axis=0).max() > 10.0
    for m in ("gfn2", "gfnff"):
        r = tasks.run(b, CalcSpec("xtb", method=m, task="sp"))
        assert r.report.get("periodic") is False
    for text in (json.loads(x) for x in inputs.read_text().splitlines()):
        xyz = np.array([[float(v) for v in ln.split()[1:4]] for ln in text.splitlines()[2:] if ln.strip()])
        assert len(xyz) == 12 and np.ptp(xyz, axis=0).max() < 5.5, xyz


def test_xtb_missing_says_how(monkeypatch, tmp_path):
    from mace.calculators import registry, tasks
    from mace.calculators.base import CalcError, CalcSpec
    monkeypatch.setenv("MACE_XTB", str(tmp_path / "nope"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    st = registry.get("xtb").status("gfn2")
    assert not st["available"] and st["commands"]
    with pytest.raises(CalcError) as e:
        tasks.run(_water_box(), CalcSpec("xtb", task="sp"))
    assert e.value.kind == "unavailable" and "conda install" in e.value.hint


def test_dftb_input(tmp_path, monkeypatch):
    """The skf method builds DFTB3 from a 3ob-like set's README; the input
    file is what ase writes (checked without running dftb+)."""
    from mace.calculators import dftb, registry
    from mace.calculators.base import CalcSpec, CalcError
    exe = tmp_path / "dftb+"
    exe.write_text("#!/bin/sh\necho '|  DFTB+ release 24.1'\n")
    exe.chmod(0o755)
    monkeypatch.setenv("MACE_DFTB", str(exe))
    dftb._STATUS.clear()
    sk = tmp_path / "set"
    sk.mkdir()
    for a in ("H", "O"):
        for b in ("H", "O"):
            (sk / f"{a}-{b}.skf").write_text("0.02 10\n")
    (sk / "README").write_text("zeta = 4.00 (gamma^h function exponent)\n\nList of all atomic Hubbard derivatives "
                              "(atomic units):\n H = -0.1857\n O = -0.1575\n\nMaximum angular momenta for calculation "
                              "input files:\n\n H = \"s\"\n O = \"p\"\n\n")
    assert dftb.read_set_readme(sk) == {"hubbard_derivs": {"H": -0.1857, "O": -0.1575}, "damping_exponent": 4.0,
                                        "max_l": {"H": "s", "O": "p"}}
    be = registry.get("dftb")
    assert be.status("skf")["status"] == "needs_skf"
    assert be.status("skf", options={"skf_dir": str(sk)})["available"]
    assert be.status("gfn2")["version"] == "24.1"
    m = _water_box()
    calc = be.calculator(m, CalcSpec("dftb", method="skf", charge=1, options={"skf_dir": str(sk)}))
    try:
        calc.write_input(m, ["energy", "forces"])
        hsd = (Path(calc.directory) / "dftb_in.hsd").read_text()
    finally:
        calc.close()
    for s in ("ThirdOrderFull = Yes", "O = -0.1575", "Exponent = 4.0", 'O = "p"', "Charge = 1.0", "ParserVersion = 8"):
        assert s in hsd, s
    with pytest.raises(CalcError):
        be.calculator(m, CalcSpec("dftb", method="skf", uhf=1, options={"skf_dir": str(sk)}))
    from ase.build import bulk
    with pytest.raises(CalcError) as e:
        be.calculator(bulk("Si"), CalcSpec("dftb", method="skf", options={"skf_dir": str(sk)}))
    assert "Si-Si" in str(e.value)



def _graphene(box=None):
    from ase import Atoms
    a = 2.46
    g = Atoms("C2", cell=[[a, 0, 0], [-a / 2, a * 3 ** 0.5 / 2, 0], [0, 0, box or 10.0]],
              scaled_positions=[[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]], pbc=[True, True, box is not None])
    return g


def test_dftb_kpoints_of_a_slab_in_a_box(tmp_path, monkeypatch):
    """A slab in a 3D box gets the k-mesh of the same slab given with pbc="T T F" (one k-point along
    the vacuum only), not the Gamma point of a molecule."""
    from mace.calculators import dftb, registry
    from mace.calculators.base import CalcSpec
    exe = tmp_path / "dftb+"
    exe.write_text("#!/bin/sh\necho '|  DFTB+ release 24.1'\n")
    exe.chmod(0o755)
    monkeypatch.setenv("MACE_DFTB", str(exe))
    dftb._STATUS.clear()
    be = registry.get("dftb")
    meshes = []
    for g in (_graphene(), _graphene(box=20.0)):
        calc = be.calculator(g, CalcSpec("dftb", method="gfn2"))
        try:
            meshes.append(tuple(calc.mace_info["kpts"]))
        finally:
            calc.close()
    assert meshes[0] == meshes[1] == (9, 9, 1), meshes
    from ase.build import molecule
    m = molecule("H2O")
    m.center(vacuum=7.0)
    m.pbc = True
    calc = be.calculator(m, CalcSpec("dftb", method="gfn2"))
    try:
        assert tuple(calc.mace_info["kpts"]) == (1, 1, 1)
    finally:
        calc.close()

@pytest.fixture
def fake_ml(tmp_path, monkeypatch):
    py = CF.F.make_fake_python(tmp_path / "venv")
    monkeypatch.setenv("MACE_MLIP_PYTHON", str(py))
    monkeypatch.setenv("FAKE_ENV_LOG", str(tmp_path / "env.json"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    return tmp_path / "env.json"


def test_mlip_auto_device_gpu0(fake_ml):
    from mace.calculators import tasks
    from mace.calculators.base import CalcSpec
    from ase.build import bulk
    r = tasks.run(bulk("Si", "diamond", a=5.5), CalcSpec("mlip", method="chgnet-0.3.0", task="opt", steps=20))
    assert r.energy is not None and "GPU 0" in r.report["device_text"]
    env = json.loads(fake_ml.read_text())
    assert env["env"]["CUDA_VISIBLE_DEVICES"] == "0" and env["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    argv = env["argv"]
    assert argv[argv.index("--device") + 1] == "auto" and argv[argv.index("--gpu-mem-fraction") + 1] == "0.5"
    assert r.provenance["model"] == "chgnet-0.3.0" and r.provenance["code_license"] == "BSD-3-Clause-LBNL"


def test_mlip_device_choices(fake_ml, monkeypatch):
    from mace.calculators import tasks
    from mace.calculators.base import CalcSpec, CalcError
    from ase.build import bulk
    tasks.run(bulk("Si"), CalcSpec("mlip", method="chgnet-0.3.0", task="sp", device="cpu"))
    env = json.loads(fake_ml.read_text())
    assert env["env"]["CUDA_VISIBLE_DEVICES"] == "" and "--gpu-mem-fraction" not in env["argv"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")        # a job's own GPUs: auto takes the first
    tasks.run(bulk("Si"), CalcSpec("mlip", method="chgnet-0.3.0", task="sp"))
    assert json.loads(fake_ml.read_text())["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    with pytest.raises(CalcError):
        tasks.run(bulk("Si"), CalcSpec("mlip", method="chgnet-0.3.0", task="sp", device="cuda:7"))


def test_mlip_missing_weights_and_a_removed_model(fake_ml):
    from mace.calculators import registry, tasks
    from mace.calculators.base import CalcSpec, CalcError
    from ase.build import bulk
    st = registry.get("mlip").status("acesuit-mace-mp-0b3-medium")
    assert st["status"] == "needs_fetch" and "mace mlip fetch acesuit-mace-mp-0b3-medium" in st["commands"]
    # MACE-OFF23 (Academic Software License) is an opt-in: before the acknowledgment it is not
    # ready and never runs, and naming it says why and how to opt in
    st = registry.get("mlip").status("acesuit-mace-off23-medium")
    assert not st["available"] and st["status"] == "needs_ack" and "academic, non-commercial use only" in st["reason"]
    assert "mace mlip fetch acesuit-mace-off23-medium --accept-academic-license" in st["hint"]
    with pytest.raises(CalcError) as e:
        tasks.run(bulk("Si"), CalcSpec("mlip", method="acesuit-mace-off23-medium", task="sp"), check_ready=False)
    assert e.value.kind == "usage" and "academic, non-commercial use only" in str(e.value)
    assert "acesuit-mace-off23-medium" not in [m.id for m in registry.get("mlip").methods()]


def test_device_mapping_auto():
    from mace.preopt.mlip_client import device_mapping
    assert device_mapping("auto", {})[:2] == ("0", "auto")
    assert device_mapping("auto", {"CUDA_VISIBLE_DEVICES": "1"})[:2] == ("1", "auto")
    assert device_mapping("auto", {"CUDA_VISIBLE_DEVICES": ""})[:2] == ("", "cpu")
    assert device_mapping("cuda:1", {})[:2] == ("1", "cuda:0")
    assert device_mapping("cpu", {})[:2] == ("", "cpu")


def test_registry_scopes_and_licenses(tmp_path):
    import copy
    from mace.preopt import registry
    assert registry.ids() == ["acesuit-mace-mpa-0-medium", "chgnet-0.3.0"]
    calc = registry.ids("calc")
    for mid in ("acesuit-mace-mp-0b3-medium", "orb-v3-conservative-inf-mpa",
                "mattersim-v1.0.0-1m", "mattersim-v1.0.0-5m"):
        assert mid in calc
    assert registry.default_id("calc") == "acesuit-mace-mpa-0-medium"
    # commercial-safe by default: every listed model's code and weights allow commercial use; MACE-OFF23
    # (ASL) is academic-only, an explicit opt-in that a plain listing leaves out
    for m in registry.load(scope=None):
        assert {m["code"]["license"], m["weights"]["license"]} <= registry.ALLOWED_LICENSES, m["id"]
    assert "acesuit-mace-off23-medium" not in registry.ids(None)
    assert registry.get("acesuit-mace-off23-medium", scope=None)["academic"] is True
    assert "known: mlip:acesuit-mace-mpa-0-medium" in registry.unknown("nope", prefix="mlip:")
    data = json.loads(registry.PATH.read_text())
    d = copy.deepcopy(data)
    m = next(x for x in d["models"] if x["id"] == "acesuit-mace-mp-0b3-medium")
    m["weights"]["license"], m["restricted"] = "ASL", True       # the 1.1.6 preview's carve-out is gone
    p = tmp_path / "asl.json"
    p.write_text(json.dumps(d))
    with pytest.raises(registry.RegistryError):
        registry.load(p)
