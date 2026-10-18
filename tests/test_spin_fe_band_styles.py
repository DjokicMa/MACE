"""Spin-split bands of a real ferromagnet draw with distinct up / down styles in mace plotting.

tests/data/spin_fe holds bcc Fe (PBE, POB-DZVP-REV2, 2.34 muB, d exchange splitting ~2 eV), run
with CRYSTAL23 on HPCC and trimmed: the BAND.DAT keeps bands 10-31 (the Fe 1s-3p core bands
are dropped, NBND and the deck's band range say so), the DOSS.DAT every third energy point.
Unlike the corpus' graphene runs, spin up and spin down here are different curves.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
FE = Path(__file__).resolve().parent / "data" / "spin_fe"
pytestmark = pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")

_PATH = re.compile(r'<path d="([^"]+)"[^>]*?style="([^"]*)"', re.S)


def _paths_by_color(svg, color):
    return [(d, st) for d, st in _PATH.findall(svg) if re.search(r"stroke:\s*" + color + r"\b", st)]


def test_fe_spin_split_bands_render_with_distinct_up_down_styles(tmp_path):
    from mace.plotting import presets
    sys.path.insert(0, str(REPO_ROOT / "Plotting"))
    import ipBANDS_V2 as ipb

    work = tmp_path / "fe"
    shutil.copytree(FE, work)
    _E, B, _Eb, Bb, *_ = ipb.read_band_data(str(work / "fe_band.BAND.DAT"), None, None)
    state, split = ipb.spin_state(B, Bb)
    assert state == "polarized" and split > 1.0, (state, split)   # a real exchange splitting (eV)

    env = {**os.environ, "MPLBACKEND": "Agg", "MACE_FOLDER_SETTINGS": "none", "MACE_PLOT_BACKEND": "matplotlib"}
    p = subprocess.run([sys.executable, str(MACE_CLI), "plotting", "--band", "fe_band.BAND.DAT",
                        "-o", str(tmp_path / "out"), "--format", "svg"],
                       cwd=str(work), capture_output=True, text=True, timeout=300, env=env)
    log = p.stdout + p.stderr
    assert p.returncode == 0 and "Traceback (most recent call last)" not in log, log[-3000:]
    svg = (tmp_path / "out" / "fe.BANDS.svg").read_text()

    up = _paths_by_color(svg, presets.SPIN_COLORS["light"]["up"])
    down = _paths_by_color(svg, presets.SPIN_COLORS["light"]["down"])
    assert len(up) >= 5 and len(down) >= 5, (len(up), len(down))
    # spin up solid, spin down dashed
    assert not any("stroke-dasharray" in st for _d, st in up)
    assert all("stroke-dasharray" in st for _d, st in down)
    # and they are different curves: the split bands do not sit on top of each other
    same = set(d for d, _ in up) & set(d for d, _ in down)
    assert len(same) < 0.2 * len(down), f"{len(same)} of {len(down)} spin-down curves equal a spin-up curve"
