"""Synthetic structures shared by the pre-optimization tests.

All hand-made: the demo CIFs of examples/gui, ase's own builders and an
FSI- anion written here from standard bond lengths (S-N 1.58, S=O 1.43,
S-F 1.57 Å, tetrahedral S) in a 15 Å cubic box. No corpus or research file.
"""
import hashlib
import os
import tempfile
import warnings
from pathlib import Path

import pytest

try:
    import numpy as np
except ImportError:          # the 3.11 CI venv: the tests that need numpy skip
    np = None

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples" / "gui" / "structures"

FSI = [("N", (7.487, 6.699, 7.507)), ("S", (8.887, 7.429, 7.507)), ("S", (6.087, 7.429, 7.507)),
       ("O", (9.125, 8.003, 8.795)), ("O", (8.886, 8.462, 6.517)), ("F", (10.018, 6.392, 7.178)),
       ("O", (5.480, 7.296, 8.795)), ("O", (5.241, 6.837, 6.517)), ("F", (6.290, 8.951, 7.178))]


def demo(name):
    import ase.io
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ase.io.read(str(EXAMPLES / name))


def fsi_box():
    from ase import Atoms
    return Atoms([s for s, _ in FSI], positions=[p for _, p in FSI], cell=[15.0] * 3, pbc=True)


def rattled_mgo(seed=7, sigma=0.03):
    a = demo("MgO.cif")
    a.positions += np.random.default_rng(seed).normal(0, sigma, a.positions.shape)
    return a


def copper():
    from ase.build import bulk
    return bulk("Cu", "fcc", a=3.61, cubic=True)


def lone_atom():
    """One argon atom in a 10 Å box: no bond to hold (copper has its metal bonds since 1.1.6)."""
    from ase import Atoms
    return Atoms("Ar", cell=[10.0, 10.0, 10.0], pbc=True)


def t0_structures():
    """The six structures the springs regression (A1) is hashed on."""
    return {"Si": demo("Si.cif"), "NaCl": demo("NaCl.cif"), "MgO": demo("MgO.cif"),
            "MgO_rattled": rattled_mgo(), "FSI_box": fsi_box(), "Cu": copper()}


def spring_hash(atoms, steps):
    return hashlib.sha256(atoms.positions.tobytes() + atoms.cell.array.tobytes() + str(steps).encode()).hexdigest()[:16]


# Spring results on the code before this feature: clean(atoms, time_limit=60),
# every run converged. Byte hashes hold for the numpy/ase they were made with;
# elsewhere the step counts are still compared. Copper (1.1.6): the springs hold
# the bonds the viewer draws, a metal's own included (it had nothing to clean).
T0_ENV = ("1.26.4", "3.25.0")
T0 = {"Si": (1, "2a4b70f288d56c50"), "NaCl": (1, "c4041029acc11dde"), "MgO": (1, "c53690fc835e7803"),
      "MgO_rattled": (43, "3bc1453193de97d4"), "FSI_box": (120, "222ba2b981f35ddd"), "Cu": (1, "782c6ba25e7e9585")}


def t0_env_matches():
    import ase
    return (np.__version__, ase.__version__) == T0_ENV


def wurtzite():
    """ZnO wurtzite with u = 0.36 (ideal 0.382): the springs move the atoms
    along c and P6_3mc stays P6_3mc, so a tier run is neither a gain nor a loss."""
    from ase.build import bulk
    return bulk("ZnO", "wurtzite", a=3.25, c=5.2, u=0.36)


FAKE_WORKER = Path(__file__).with_name("fake_mlip_worker.py")

_FAKE_PYTHON = r'''#!{exe}
# A stand-in ML interpreter for the tests: answers the environment probe as an
# environment with torch, ase, mace-torch and chgnet, and runs the fake worker
# in place of mlip_worker.py (same argv after the script path).
import json, os, sys
args = sys.argv[1:]
if "-c" in args:
    if os.environ.get("FAKE_PROBE") == "fail":
        sys.stderr.write("ImportError: libtorch.so: cannot open shared object file\n")
        sys.exit(1)
    if os.environ.get("FAKE_PROBE") == "hang":
        import time; time.sleep(60)
    ok = {{"ok": True, "version": "0.0-fake"}}
    print("MACE_PROBE_JSON " + json.dumps({{"python": "3.12.2", "prefix": "/fake", "packages": {{
        "torch": ok, "ase": ok, "mace-torch": ok, "chgnet": ok}}}}))
    sys.exit(0)
i = args.index("-I") if "-I" in args else -1
rest = args[i + 2:] if i >= 0 else args[1:]
os.execv(sys.executable, [sys.executable, "-I", {worker!r}] + rest)
'''


def make_fake_python(directory):
    """``directory/bin/python``: the fake ML interpreter (see _FAKE_PYTHON)."""
    import os
    import sys
    b = Path(directory) / "bin"
    b.mkdir(parents=True, exist_ok=True)
    p = b / "python"
    p.write_text(_FAKE_PYTHON.format(exe=sys.executable, worker=str(FAKE_WORKER)))
    p.chmod(0o755)
    return p


# mlip_install's tests: a fake base Python and its venv's python. The venv's python: pip logs its argv (and whether PYTHONPATH reached it); the probe answers
VENV_PY = r'''#!{exe}
import json, os, sys, time
args = sys.argv[1:]
log = os.environ.get("FAKE_LOG")
if "-c" in args:
    ok = {{"ok": True, "version": "1.0-fake"}}
    print("MACE_PROBE_JSON " + json.dumps({{"python": "3.12.2", "prefix": sys.prefix, "packages": {{
        "torch": ok, "ase": ok, "mace-torch": ok, "chgnet": ok}}}}))
    sys.exit(0)
if args[:2] == ["-m", "pip"]:
    with open(log, "a") as fh:
        fh.write(json.dumps({{"argv": args[2:], "pythonpath": "PYTHONPATH" in os.environ}}) + "\n")
    print("Collecting things (fake pip)")
    fail = os.environ.get("FAKE_PIP_FAIL")
    if fail and fail in " ".join(args[2:]):
        print("ERROR: fake failure"); sys.exit(1)
    nap = os.environ.get("FAKE_PIP_SLEEP")
    if nap and nap.split(":")[0] in " ".join(args[2:]):
        time.sleep(float(nap.split(":")[1]))
    sys.exit(0)
sys.exit(3)
'''

# the base python: answers the version check and makes a venv-shaped folder
BASE_PY = r'''#!{exe}
import os, sys
args = sys.argv[1:]
if "-c" in args:
    print("3 12 2"); sys.exit(0)
if args[:2] == ["-m", "venv"]:
    d = args[2]
    os.makedirs(os.path.join(d, "bin"), exist_ok=True)
    open(os.path.join(d, "pyvenv.cfg"), "w").write("home = /fake\ninclude-system-site-packages = false\n")
    p = os.path.join(d, "bin", "python")
    open(p, "w").write(open(os.environ["FAKE_VENV_PY"]).read())
    os.chmod(p, 0o755)
    print("created", d); sys.exit(0)
sys.exit(3)
'''


def make_fake_base_python(directory, log):
    """``directory/python3``: a fake base Python for mlip_install (its venv's
    pip logs to ``log``); returns (path, the environment it needs)."""
    import sys
    b = Path(directory)
    b.mkdir(parents=True, exist_ok=True)
    (b / "venv_py").write_text(VENV_PY.format(exe=sys.executable))
    base = b / "python3"
    base.write_text(BASE_PY.format(exe=sys.executable))
    base.chmod(0o755)
    return base, {"FAKE_LOG": str(log), "FAKE_VENV_PY": str(b / "venv_py")}


def load_fake_calc():
    """The fake worker's harmonic calculator, in this process."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fake_mlip_worker_mod", FAKE_WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._calc_class()()


# --- the sandbox every pre-optimization test runs in -----------------------------------
# Each test module does `from preopt_fixtures import preopt_sandbox  # noqa: F401`
# (a conftest.py here would shadow tests/conftest.py's module name).

REAL = {k: os.environ.get(k) for k in ("MACE_XTB", "MACE_MLIP_PYTHON", "MACE_MLIP_DIR")}


@pytest.fixture(autouse=True)
def preopt_sandbox(tmp_path, monkeypatch):
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    for k in ("MACE_XTB", "MACE_MLIP_PYTHON", "MACE_MLIP_DIR", "MACE_MLIP_NO_DOWNLOAD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    try:
        from mace.preopt import detect
        detect._XTB_CACHE.clear()
        detect._MEM.clear()
    except ImportError:
        pass
    yield tmp
    root = tmp / f"mace-preopt-{os.getuid()}"
    left = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
    assert not left, f"temp directories left behind: {left}"
