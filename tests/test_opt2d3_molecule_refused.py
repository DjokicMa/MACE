"""opt2d3 writes no BAND, DOSS or TRANSPORT deck for a molecule.

CRYSTAL23's manual says of BAND "does not run for molecules!! (0D)" (p.309)
and of DOSS "do not run for molecules!" (p.320); BOLTZTRA's transport is
computed from the bands. opt2d3 wrote a triclinic k-path BAND deck for a
MOLECULE output. It now refuses before asking anything; CHARGE and
POTENTIAL (ECH3/POT3 work for molecules) are unchanged. Run through mace_cli
as the user runs it (the GUI's Setup and Batch Setup run this command too).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

pytest.importorskip("numpy")

PARENT = REPO_ROOT / "tests" / "fixtures" / "gui_parity" / "water_opt.out"


def _run(tmp_path, calc_type):
    d = tmp_path / "run"
    d.mkdir()
    shutil.copy(PARENT, d / PARENT.name)
    (d / PARENT.with_suffix(".f9").name).write_text("wavefunction placeholder\n")
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH"):
        env.pop(k, None)
    env["MACE_NO_BANNER"] = "1"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3",
                        "--input", str(d / PARENT.name), "--calc-type", calc_type],
                       cwd=str(d), capture_output=True, text=True, env=env, timeout=180, input="\n" * 60)
    return d, r


@pytest.mark.parametrize("calc_type", ["BAND", "DOSS", "TRANSPORT"])
def test_no_band_doss_or_transport_deck_for_a_molecule(tmp_path, calc_type):
    d, r = _run(tmp_path, calc_type)
    text = r.stdout + r.stderr
    assert r.returncode != 0, text[-2000:]
    assert "requires a periodic structure" in text and "molecule" in text, text[-2000:]
    assert not list(d.glob("*.d3")) and not (d / f"water_opt_{calc_type.lower()}.f9").exists(), sorted(p.name for p in d.iterdir())


def test_charge_deck_for_a_molecule_still_written(tmp_path):
    d, r = _run(tmp_path, "CHARGE")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert list(d.glob("*.d3"))
