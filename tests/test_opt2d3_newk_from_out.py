"""opt2d3 DOSS/TRANSPORT keep the parent's k-mesh when only the .out is there.

HPCC re-test (1.1.5): with only MgO_sp.out and .f9 in the directory, the
Enter-through DOSS deck had 'NEWK 8 16' although the SCF ran on SHRINK 10 20
(the .out prints 'SHRINK. FACT.(MONKH.) 10 10 10' and 'SHRINKING
FACTOR(GILAT NET) 20'); with MgO_sp.d12 copied in it had '10 20', as the
workflow deck did. The NEWK now comes from the .out's SHRINK when there is no
.d12, so the deck is the same either way. Run through mace_cli as the user
runs it.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

pytest.importorskip("numpy")

DATA = Path(__file__).parent / "data"
PARENTS = {
    "lih_3d_shrink_is": DATA / "label_band" / "lih_p1bar.out",
    "hbn_slab": DATA / "label_band" / "hbn_p1_slab.out",
    "graphene_shrink_0": DATA / "low_dim_groups" / "graphene_lg80.out",
    "polyyne_1d": DATA / "low_dim_groups" / "polyyne_rg4.out",
}


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH"):
        env.pop(k, None)
    env["MACE_NO_BANNER"] = "1"
    return env


def _deck(tmp_path, src, calc_type, with_d12):
    d = tmp_path / ("with_d12" if with_d12 else "out_only")
    d.mkdir()
    shutil.copy(src, d / src.name)
    if with_d12:
        shutil.copy(src.with_suffix(".d12"), d / src.with_suffix(".d12").name)
    (d / src.with_suffix(".f9").name).write_text("wavefunction placeholder\n")
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3",
                        "--input", str(d / src.name), "--calc-type", calc_type],
                       cwd=str(d), input="\n" * 40, capture_output=True, text=True,
                       env=_env(), timeout=180)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    (deck,) = d.glob(f"*_{calc_type.lower()}.d3")
    return deck.read_text(), r.stdout + r.stderr


def _newk(deck):
    lines = deck.splitlines()
    return lines[lines.index("NEWK") + 1]


@pytest.mark.parametrize("calc_type", ["DOSS", "TRANSPORT"])
@pytest.mark.parametrize("parent", sorted(PARENTS))
def test_out_only_deck_is_the_deck_with_the_d12(tmp_path, parent, calc_type):
    with_d12, _ = _deck(tmp_path, PARENTS[parent], calc_type, True)
    out_only, _ = _deck(tmp_path, PARENTS[parent], calc_type, False)
    assert _newk(out_only) == _newk(with_d12)
    assert out_only == with_d12


def test_mgo_like_mesh_is_read_from_the_out(tmp_path):
    """The HPCC case's numbers: SHRINK 10 10 10 / Gilat 20 -> NEWK 10 20."""
    import re
    src = PARENTS["lih_3d_shrink_is"]
    text, n = re.subn(
        r" SHRINK\. FACT\.\(MONKH\.\)\s+6\s+6\s+6 .*",
        " SHRINK. FACT.(MONKH.)   10 10 10  NUMBER OF K POINTS IN THE IBZ     47\n"
        " SHRINKING FACTOR(GILAT NET)   20  NUMBER OF K POINTS(GILAT NET)    256",
        src.read_text())
    assert n == 1
    staged = tmp_path / "src"
    staged.mkdir()
    (staged / src.name).write_text(text)
    deck, log = _deck(tmp_path, staged / src.name, "DOSS", False)
    assert _newk(deck) == "10 20", deck


def test_no_mesh_anywhere_says_so(tmp_path):
    """No .d12 and no SHRINK in the .out: the fallback mesh is named, not silent."""
    src = PARENTS["lih_3d_shrink_is"]
    staged = tmp_path / "src"
    staged.mkdir()
    text = "\n".join(l for l in src.read_text().splitlines()
                     if "SHRINK" not in l)
    (staged / src.name).write_text(text)
    deck, log = _deck(tmp_path, staged / src.name, "DOSS", False)
    assert _newk(deck) == "8 16"
    assert "no SHRINK" in log
