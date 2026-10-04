"""opt2d3 leaves no <base>_<type>.f9 behind when it writes no deck.

HPCC re-test (1.1.5): `opt2d3 --input mgo_sp.out --calc-type BAND < /dev/null`
stopped with "The input ended before every question was answered; no D3 file
was written", yet mgo_band.f9 (a full copy of the wavefunction) was left in
the directory; the same for DOSS. The wavefunction is now copied only once
the deck is written. Run through mace_cli as the user runs it.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

pytest.importorskip("numpy")

PARENT = Path(__file__).parent / "data" / "label_band" / "lih_p1bar.out"


def _run(tmp_path, calc_type, stdin):
    d = tmp_path / "run"
    d.mkdir()
    shutil.copy(PARENT, d / PARENT.name)
    (d / PARENT.with_suffix(".f9").name).write_text("wavefunction placeholder\n")
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH"):
        env.pop(k, None)
    env["MACE_NO_BANNER"] = "1"
    kw = {"stdin": subprocess.DEVNULL} if stdin is None else {"input": stdin}
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3",
                        "--input", str(d / PARENT.name), "--calc-type", calc_type],
                       cwd=str(d), capture_output=True, text=True, env=env, timeout=180, **kw)
    return d, r


@pytest.mark.parametrize("calc_type", ["BAND", "DOSS"])
def test_input_that_ends_early_leaves_no_wavefunction_copy(tmp_path, calc_type):
    d, r = _run(tmp_path, calc_type, None)
    text = r.stdout + r.stderr
    assert "no D3 file was written" in text, text
    stem = PARENT.stem
    assert not (d / f"{stem}_{calc_type.lower()}.d3").exists()
    assert not (d / f"{stem}_{calc_type.lower()}.f9").exists(), sorted(p.name for p in d.iterdir())


@pytest.mark.parametrize("calc_type", ["BAND", "DOSS"])
def test_written_deck_still_gets_its_wavefunction(tmp_path, calc_type):
    d, r = _run(tmp_path, calc_type, "\n" * 40)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    stem = PARENT.stem
    assert (d / f"{stem}_{calc_type.lower()}.d3").exists()
    assert (d / f"{stem}_{calc_type.lower()}.f9").read_text() == "wavefunction placeholder\n"
