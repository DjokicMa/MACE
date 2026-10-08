"""ipDOS_V2's DOSS readers, lifted out of ipDOS() so other code can reuse them.

The plot ipDOS() draws is unchanged: it is compared, pixel for pixel, with the
plot the pre-refactor ipDOS_V2.py (the 1.1.5 base before the GUI, daa5468e) draws
from the same synthetic files, when git can show that file.
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT

PLOTTING = REPO_ROOT / "Plotting"
BASE = "daa5468e"


def _load(path, name):
    if str(PLOTTING) not in sys.path:
        sys.path.insert(0, str(PLOTTING))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_doss(directory, material="mgo", n_e=60, spin_blocks=2):
    """A small two-element DOSS run: .d3 with 3 projections + total."""
    d3 = ("NEWK\n8 8\n1 0\nDOSS\n3 400 -1 -1 2 14 0\n-0.5 0.5\n"
          "4  1 2 3 4  #MG all\n4  5 6 7 8  #O S\n4  9 10 11 12  #O P\nEND\n")
    (directory / f"{material}_doss.d3").write_text(d3)
    e = np.linspace(-0.3, 0.3, n_e)
    lines = [f"# NEPTS {n_e:5d} NPROJ     4 NSPIN     {spin_blocks}", "#",
             '@ XAXIS LABEL "E-EFERMI (HARTREE)"']
    for block in range(spin_blocks):
        sign = 1 if block == 0 else -1
        for k, x in enumerate(e):
            cols = [x] + [sign * (np.sin(3 * x + j) ** 2 * 10 + j) for j in range(4)]
            lines.append(" ".join(f"{c: .4E}" for c in cols))
        lines += ["# EFERMI (HARTREE)   -0.08011", "&"]
    (directory / f"{material}_doss.DOSS.DAT").write_text("\n".join(lines) + "\n")
    return material


def test_lifted_readers_return_the_columns():
    ip = _load(PLOTTING / "ipDOS_V2.py", "ipDOS_V2_now")
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    m = write_doss(d)
    raw = ip.read_doss_projection_labels(str(d / f"{m}_doss.d3"))
    assert [ip.doss_label(r) for r in raw] == ["Mg ", "O (s)", "O (p)", "Total"]
    cols = ip.read_doss_columns(str(d / f"{m}_doss.DOSS.DAT"), 1 + len(raw), 0)
    assert len(cols) == 5 and len(cols[0]) == 120          # both spin blocks, in order
    assert cols[0][0] == pytest.approx(-0.3 * 27.2114)
    assert ip.vacuum_shift(str(d / "none.DAT"), str(d / "none.out")) == 0


def test_ipdos_plot_is_unchanged_by_the_refactor(tmp_path):
    try:
        old = subprocess.run(["git", "show", f"{BASE}:Plotting/ipDOS_V2.py"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available")
    if old.returncode != 0:
        pytest.skip(f"{BASE} is not in this clone")
    (tmp_path / "old.py").write_text(old.stdout)
    pngs = []
    for label, path in (("old", tmp_path / "old.py"), ("new", PLOTTING / "ipDOS_V2.py")):
        work = tmp_path / label
        work.mkdir()
        m = write_doss(work)
        mod = _load(path, f"ipDOS_V2_{label}")
        cwd = os.getcwd()
        os.chdir(work)
        try:
            mod.ipDOS(m, -5, 5, "orbital", formats="png", output_dir=str(work))
        finally:
            os.chdir(cwd)
            import matplotlib.pyplot as plt
            plt.close("all")
        found = sorted(work.glob("*.png"))
        assert found, label
        pngs.append(found[0].read_bytes())
    assert pngs[0] == pngs[1]
