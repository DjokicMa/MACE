"""A backend that fails at run time (xtb's SCF does not converge, DFTB+ stops,
the ML worker crashes) is a message from `mace calc`, not a Python traceback:
what failed, the backend's own words and the folder its log files were kept
in, exit code 1; `--debug` shows the traceback. A MACE bug is never turned
into such a message. Stand-in programs play the failing backends."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ase")
pytest.importorskip("spglib")
import calc_fixtures as CF  # noqa: E402
from calc_fixtures import preopt_sandbox  # noqa: E402,F401

CLI = CF.REPO / "mace_cli"
WATER = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n"

FAILING_XTB = r'''#!{exe}
import sys
if "--version" in sys.argv:
    print("      * xtb version 6.7.1 (fake) compiled by x on 2024")
    sys.exit(0)
print("[ERROR] Program stopped due to fatal error")
print("  -2- scc: Self consistent charge iterator did not converge")
print("  -1- singlepoint: Electronic structure method terminated")
sys.exit(1)
'''

FAILING_DFTB = r'''#!/bin/sh
if [ -f dftb_in.hsd ]; then
  echo "ERROR!"
  echo "-> SCC is NOT converged, maximal SCC iterations exceeded"
  exit 1
fi
echo '|  DFTB+ release 24.1'
'''


def mace(*args, cwd=None, env=None, timeout=300):
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_XTB", "MACE_DFTB", "MACE_MLIP_PYTHON")}
    e.update(env or {})
    p = subprocess.run([sys.executable, str(CLI), "--no-banner", *args], cwd=cwd, env=e, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def _program(tmp_path, name, text):
    p = tmp_path / "bin" / name
    p.parent.mkdir(exist_ok=True)
    p.write_text(text.format(exe=sys.executable) if text.startswith("#!{exe}") else text)
    p.chmod(0o755)
    return p


def _kept(err):
    """The log folder the message names; removed here (the sandbox must end empty)."""
    m = re.search(r"Its log files were kept in (\S+)", err)
    assert m, err
    d = Path(m.group(1))
    assert d.is_dir() and d.parent.name.startswith("mace-preopt-"), d
    names = sorted(p.name for p in d.iterdir())
    shutil.rmtree(d)
    return names


def _clean(err):
    assert "Traceback" not in err, err
    assert "Error:" not in err, err          # no exception class name either (GFNFFError: ..., MlipError: ...)


def test_xtb_scf_failure_is_a_message(tmp_path):
    xtb = _program(tmp_path, "xtb", FAILING_XTB)
    (tmp_path / "water.xyz").write_text(WATER)
    rc, out, err = mace("calc", "sp", str(tmp_path / "water.xyz"), "-b", "xtb", "-m", "gfn2",
                        env={"MACE_XTB": str(xtb)})
    assert rc == 1, (out, err)
    _clean(err)
    assert "mace calc: GFN2-xTB failed during the single point: GFN2-xTB failed: xtb exit code 1" in err, err
    assert "Self consistent charge iterator did not converge" in err
    assert "xtb.out" in _kept(err)
    assert not list(tmp_path.glob("water_xtb-*"))           # nothing written
    rc, _, err = mace("calc", "sp", str(tmp_path / "water.xyz"), "-b", "xtb", "-m", "gfn2", "--debug",
                      env={"MACE_XTB": str(xtb)})
    assert rc == 1
    assert "Traceback" in err and "GFNFFError" in err and "mace calc: GFN2-xTB failed" in err
    _kept(err)


def test_dftb_failure_is_a_message(tmp_path):
    exe = _program(tmp_path, "dftb+", FAILING_DFTB)
    (tmp_path / "water.xyz").write_text(WATER)
    rc, out, err = mace("calc", "sp", str(tmp_path / "water.xyz"), "-b", "dftb", "-m", "gfn2",
                        env={"MACE_DFTB": str(exe)})
    assert rc == 1, (out, err)
    _clean(err)
    assert ("GFN2-xTB Hamiltonian in DFTB+ failed during the single point: DFTB+ stopped (exit code 1): "
            "SCC is NOT converged, maximal SCC iterations exceeded") in err, err
    assert {"dftb.out", "dftb_in.hsd"} <= set(_kept(err))


def test_ml_worker_crash_is_a_message(tmp_path):
    py = CF.F.make_fake_python(tmp_path / "venv")
    cif = tmp_path / "NaCl.cif"
    shutil.copy(CF.F.EXAMPLES / "NaCl.cif", cif)
    rc, out, err = mace("calc", "sp", str(cif), "-b", "mlip", "-m", "chgnet-0.3.0", "--device", "cpu",
                        env={"MACE_MLIP_PYTHON": str(py), "FAKE_MODE": "crash"})
    assert rc == 1, (out, err)
    _clean(err)
    assert "CHGNet 0.3.0 failed during the single point: crash: the ML worker exited (code 9) during a call" in err
    assert _kept(err) == ["worker.log"]


def test_a_mace_bug_stays_a_traceback(monkeypatch):
    """Only the backends' own failures become messages: anything else propagates unchanged."""
    from ase.build import bulk
    from mace.calculators import base, tasks
    from mace.preopt.gfnff import GFNFFError

    def bug(*a, **k):
        raise KeyError("a MACE bug")
    monkeypatch.setitem(tasks.DRIVERS, "sp", bug)
    with pytest.raises(KeyError):
        tasks.run(bulk("Si"), base.CalcSpec("uff", task="sp"))

    def fails(*a, **k):
        raise GFNFFError("SCF did not converge")
    monkeypatch.setitem(tasks.DRIVERS, "sp", fails)
    with pytest.raises(base.CalcError) as e:
        tasks.run(bulk("Si"), base.CalcSpec("uff", task="sp"))
    assert e.value.kind == "failed" and "failed during the single point: SCF did not converge" in str(e.value)
    assert isinstance(e.value.__cause__, GFNFFError)
