"""ipDOS_V2's DOSS readers, lifted out of ipDOS() so other code can reuse them.

The plot ipDOS() draws is unchanged: it is compared with the plot the
pre-refactor ipDOS_V2.py (the 1.1.5 base before the GUI, daa5468e) draws from
the same synthetic files, artist by artist (the curves, fills, limits, ticks,
labels and legend entries). Pixel for pixel it is not the same since the legend
moved outside the axes (the band-gap and band-edge composition notes heading it). That file is frozen in
tests/frozen, so the comparison also runs in a shallow clone (CI) that lacks
the base commit.
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
FROZEN_BASE = REPO_ROOT / "tests" / "frozen" / "ipDOS_V2.base.txt"   # Plotting/ipDOS_V2.py at BASE


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


def _artists(fig):
    """What the DOS plot shows, legend placement and band-gap note aside."""
    ax = fig.axes[0]
    gap = ("$E_g", "Conductive", "Semimetal", "VB: ", "CB: ", "$E_F$: ")      # notes now heading the legend
    return {
        "lim": (ax.get_xlim(), ax.get_ylim()),
        "lines": [(ln.get_label(), ln.get_color(), ln.get_linestyle(), np.asarray(ln.get_xydata()).round(9).tolist())
                  for ln in ax.get_lines()],
        "fills": [[np.asarray(p.vertices).round(9).tolist() for p in c.get_paths()] for c in ax.collections],
        "ticks": ([t.get_text() for t in ax.get_xticklabels()], ax.get_xticks().tolist(), ax.get_yticks().tolist()),
        "labels": (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()),
        "texts": [(t.get_text(), t.get_position()) for t in ax.texts if not t.get_text().startswith(gap)],
        "legend": [t.get_text() for t in ax.get_legend().get_texts()],
    }


def test_ipdos_plot_is_unchanged_by_the_refactor(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    (tmp_path / "old.py").write_text(FROZEN_BASE.read_text())
    drawn = []
    for label, path in (("old", tmp_path / "old.py"), ("new", PLOTTING / "ipDOS_V2.py")):
        work = tmp_path / label
        work.mkdir()
        m = write_doss(work)
        mod = _load(path, f"ipDOS_V2_{label}")
        saved = []
        monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: saved.append(self))
        monkeypatch.setattr(mod.plt, "close", lambda *a, **k: None)
        cwd = os.getcwd()
        os.chdir(work)
        try:
            mod.ipDOS(m, -5, 5, "orbital", formats="png", output_dir=str(work))
        finally:
            os.chdir(cwd)
        assert saved, label
        drawn.append(_artists(saved[-1]))
        plt.close("all")
    # the intended changes since (1.1.6): the Fermi line is dotted, as in the band plot and the GUI; a
    # alpha = beta DOS is one-sided (B2: the GUI's rule), each curve the spin sum alpha + |beta| (both blocks
    # of an unrestricted run hold one spin each, so the one-sided curve is the total); the projections
    # are named and colored as in the GUI ("O p", T7); the total is an ink line over a light fill (T5); the
    # figure is 3.5 x 5 in without a title (T8). The data drawn are the same.
    old, new = drawn
    old_curves = {lab.strip(): xy for lab, _c, _ls, xy in old["lines"] if not lab.startswith(("$E_F", "_"))}
    new_curves = {lab: xy for lab, _c, _ls, xy in new["lines"] if lab and lab != "Total" and not lab.startswith(("$E_F", "_"))}
    # T5 / 1.1.6 N1: a closed-shell total is an ink line (the GUI's plot ink) over a light neutral fill
    from mace.plotting.plotly_bands import tokens
    assert [c for lab, c, _ls, _xy in new["lines"] if lab == "Total"] == [tokens("default")["--plot-ink"]]
    names = {"O (s)": "O s", "O (p)": "O p"}
    assert set(new_curves) == {names[k] for k in old_curves}, (old_curves.keys(), new_curves.keys())
    for k, xy in old_curves.items():
        half = len(xy) // 2
        summed = [[round(a[0] + abs(b[0]), 9), a[1]] for a, b in zip(xy[:half], xy[half:])]
        assert np.allclose(new_curves[names[k]], summed, atol=1e-8), k   # alpha + |beta|: one-sided total
    # one-sided: from 0 to the largest drawn (summed) DOS in the window plus 8% (1.1.6, the GUI's range)
    lo, hi = new["lim"][1]
    top = max(abs(x) for lab, _c, _ls, xy in new["lines"] if lab and not lab.startswith(("$E_F", "_"))
              for x, y in xy if lo <= y <= hi)
    assert new["lim"][0][0] == 0 and new["lim"][0][1] == pytest.approx(1.08 * top) and new["lim"][1] == old["lim"][1]
    # 1.1.6: the energy title plain, not bold (the GUI's), and the DOS unit "states/eV"
    assert new["labels"][2] == "" and new["labels"][1] == "$E - E_F$ (eV)"
    assert [lab for lab, _c, ls, _xy in new["lines"] if lab.startswith("$E_F")] == ["$E_F$"]


def test_frozen_base_is_the_base_commit_file():
    """tests/frozen/ipDOS_V2.base.txt is Plotting/ipDOS_V2.py at BASE, when git has it."""
    try:
        old = subprocess.run(["git", "show", f"{BASE}:Plotting/ipDOS_V2.py"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available")
    if old.returncode != 0:
        pytest.skip(f"{BASE} is not in this clone (shallow); the frozen copy is used above")
    assert FROZEN_BASE.read_text() == old.stdout
