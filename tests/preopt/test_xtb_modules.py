"""xtb found through the machine's module system (mace/utils/envmodules.py),
with a fake module system: MACE_XTB -> PATH (6.7+) -> the newest module 6.7+,
loaded automatically for xtb's process only; computed hints."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import preopt_fixtures as F
from preopt_fixtures import preopt_sandbox  # noqa: F401  (autouse)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envmodules_fake as fake  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
# the real corpus CIF when the gitignored test/ folder is present, else the shipped example
CORPUS_CIF = Path(__file__).resolve().parents[2] / "test" / "CIFs" / "1_dia_opt_rev1.cif"

# a module's xtb that runs only with its module's library path (like a real module build)
NEEDS_LIB = ('case ":$LD_LIBRARY_PATH:" in *"xtb_6.7.1-new/lib:"*) ;; *) echo "xtb: error while loading shared libraries" >&2; '
             'exit 127 ;; esac\n')


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    from mace.utils import envmodules
    envmodules._MEM.clear()
    yield
    envmodules._MEM.clear()


def _fake(tmp_path, monkeypatch, mods, programs, conflicts=()):
    env = fake.make(tmp_path, avail=mods, programs=programs, conflicts=conflicts)
    fake.apply(monkeypatch, env)
    monkeypatch.setenv("XDG_CACHE_HOME", env["XDG_CACHE_HOME"])
    return env


def test_no_path_xtb_loads_the_module(tmp_path, monkeypatch):
    from mace.preopt import detect
    _fake(tmp_path, monkeypatch, ["xtb/6.6.1-old", "xtb/6.7.1-new"],
          {"xtb/6.7.1-new": ("xtb", NEEDS_LIB + fake.XTB_671), "xtb/6.6.1-old": ("xtb", fake.XTB_661)},
          conflicts=["xtb/6.7.1-new"])
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    st = detect.xtb_status()
    assert st["status"] == "ready" and st["source"] == "module" and st["module"] == "xtb/6.7.1-new", st
    assert "xtb_6.7.1-new/lib" in st["env"]["LD_LIBRARY_PATH"]
    assert "xtb_6.7.1-new/lib" not in os.environ.get("LD_LIBRARY_PATH", "")
    assert detect.xtb_path() == st["path"]


def test_path_too_old_then_module(tmp_path, monkeypatch):
    from mace.preopt import detect
    _fake(tmp_path, monkeypatch, ["xtb/6.7.1-new"], {"xtb/6.7.1-new": ("xtb", fake.XTB_671)})
    old = tmp_path / "oldbin"
    old.mkdir()
    (old / "xtb").write_text("#!/bin/sh\n" + fake.XTB_661)
    (old / "xtb").chmod(0o755)
    monkeypatch.setenv("PATH", f"{old}:/usr/bin:/bin")
    st = detect.xtb_status()
    assert st["status"] == "ready" and st["source"] == "module" and st["version"] == "6.7.1"


def test_mace_xtb_always_wins(tmp_path, monkeypatch):
    from mace.preopt import detect
    env = _fake(tmp_path, monkeypatch, ["xtb/6.7.1-new"], {"xtb/6.7.1-new": ("xtb", fake.XTB_671)})
    mine = tmp_path / "mine"
    mine.write_text("#!/bin/sh\n" + fake.XTB_661)
    mine.chmod(0o755)
    monkeypatch.setenv("MACE_XTB", str(mine))
    st = detect.xtb_status()
    assert st["source"] == "MACE_XTB" and st["status"] == "too_old" and fake.calls(env) == []


def test_only_too_old_modules_give_a_computed_hint(tmp_path, monkeypatch):
    from mace.preopt import detect
    _fake(tmp_path, monkeypatch, ["xtb/6.6.1-old"], {"xtb/6.6.1-old": ("xtb", fake.XTB_661)})
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    st = detect.xtb_status()
    assert st["status"] == "not_found"
    assert "the newest xtb module here is xtb/6.6.1-old" in st["hint"]
    assert "module load" not in st["hint"]            # MACE loads it itself; no command that may fail
    assert 'conda install -c conda-forge "xtb>=6.7"' in st["hint"]


def test_override_none_turns_discovery_off(tmp_path, monkeypatch):
    from mace.preopt import detect
    env = _fake(tmp_path, monkeypatch, ["xtb/6.7.1-new"], {"xtb/6.7.1-new": ("xtb", fake.XTB_671)})
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("MACE_XTB_MODULE", "none")
    st = detect.xtb_status()
    assert st["status"] == "not_found" and fake.calls(env) == []


FAKE_GFNFF = r'''
case ":$LD_LIBRARY_PATH:" in *"xtb_6.7.1-new/lib:"*) ;; *) echo "missing libs" >&2; exit 127 ;; esac
if [ "$1" = --version ]; then echo "      * xtb version 6.7.1 (edcfbbe) compiled by test"; exit 0; fi
n=$(sed -n 7p POSCAR | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')
echo "          | TOTAL ENERGY              -5.000000000000 Eh   |"
{ echo '$grad'; echo '  cycle =      1    SCF energy =    -5.0  |dE/xyz| =  0.0'
  i=0; while [ $i -lt $n ]; do echo '  0.0  0.0  0.0  x'; i=$((i+1)); done
  i=0; while [ $i -lt $n ]; do echo '  0.0D+00  0.0D+00  0.0D+00'; i=$((i+1)); done
  echo '$end'; } > gradient
{ echo '$gradlatt'; echo '  cycle =      1    SCF energy =    -5.0  |dE/xyz| =  0.0'
  echo '  0 0 0'; echo '  0 0 0'; echo '  0 0 0'
  echo '  0.0D+00  0.0D+00  0.0D+00'; echo '  0.0D+00  0.0D+00  0.0D+00'; echo '  0.0D+00  0.0D+00  0.0D+00'
  echo '$end'; } > gradlatt
'''


def test_real_invocation_preopt_gfnff_uses_the_module(tmp_path, monkeypatch):
    """mace_cli preopt --method gfnff as a subprocess, no xtb on PATH: the
    module is found and loaded for xtb; the provenance names it."""
    pytest.importorskip("ase")
    pytest.importorskip("spglib")
    src = CORPUS_CIF if CORPUS_CIF.is_file() else F.EXAMPLES / "Si.cif"
    env = _fake(tmp_path, monkeypatch, ["xtb/6.7.1-new"], {"xtb/6.7.1-new": ("xtb", FAKE_GFNFF)},
                conflicts=["xtb/6.7.1-new"])
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(src, work / "C.cif")
    e = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    e["PATH"] = "/usr/bin:/bin"
    r = subprocess.run([sys.executable, str(F.REPO / "mace_cli"), "--no-banner", "preopt", "C.cif",
                        "--method", "gfnff", "--no-fallback"], cwd=work, env=e, capture_output=True, text=True,
                       timeout=300, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Used xtb 6.7.1 (module xtb/6.7.1-new, loaded automatically)" in r.stdout
    prov = json.loads((work / "C_preopt.json").read_text())
    assert prov["method_ran"] == "gfnff" and prov["code_module"] == "xtb/6.7.1-new"
    assert any(c.startswith("load xtb/6.7.1-new") for c in fake.calls(env))
    # a second run hits the module cache: no module command at all
    n = len(fake.calls(env))
    r = subprocess.run([sys.executable, str(F.REPO / "mace_cli"), "--no-banner", "preopt", "C.cif",
                        "--method", "gfnff", "--no-fallback", "--force"], cwd=work, env=e, capture_output=True,
                       text=True, timeout=300, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    assert len(fake.calls(env)) == n
