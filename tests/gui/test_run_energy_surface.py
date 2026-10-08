"""The OPT energy chart's first point on the optimizer's energy surface
(PLAN_analysis C0).

CRYSTAL prints point 1's ``TOTAL ENERGY(DFT)`` without the dispersion/gCP
correction and the corrected total on a later ``TOTAL ENERGY + DISP (+ GCP)
(AU)`` line; from point 2 on, ``TOTAL ENERGY(DFT)`` already includes it.
outindex v3 adds ``opt_energies_au`` (that surface; the final frame takes
the OPT END energy) and keeps ``energies_au`` unchanged.

Oracle for every real OPT copy: point 2's printed ``DE (AU)`` is relative to
point 1 (CRYSTAL's only earlier point), so ``opt[point 2] - opt[point 1]``
equals it within its printed digits. (Later DE values are relative to
CRYSTAL's current reference geometry, which moves, so they are not an oracle
for consecutive differences.)"""
import re
import sys
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import runcorpus as RC

sys.path[:0] = [p for p in (str(H.REPO_ROOT), str(H.REPO_ROOT / "mace"), str(H.REPO_ROOT / "Crystal_d3")) if p not in sys.path]


def _outindex():
    from mace.gui import outindex
    return outindex


def test_dia_opt_energies(tmp_path):
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    (p,) = RC.copy_into(tmp_path, RC.DIA_OPT)
    oi = _outindex()
    idx = oi.build(p)
    want = [-76.217159957139, -76.217205447083, -76.217215565143, -76.217187737807, -76.21718773781]
    got = idx.opt_energies()
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert abs(g - w) < 1e-10
    # energies_au is v2's answer, unchanged
    assert idx.energies() == [-76.194318972376, -76.217205447083, -76.217215565143, -76.217187737807, None]
    s = oi.summary(idx, p.name)
    assert s["energies_au"] == idx.energies() and s["index_version"] == oi.VERSION >= 3
    # the summary sends the corrected values only where they differ (the client expands them)
    assert s["opt_energies_at"] == {"0": got[0], "4": got[4]}


_DE = re.compile(r"TOTAL ENERGY\((?:DFT|HF)\)\(AU\)\(\s*\d+\)\s+(\S+)\s+DE \(AU\)\s+(\S+)")


def test_point_two_de_matches_the_corrected_first_point(tmp_path):
    oi = _outindex()
    outs = RC.corpus_outs("OPT")
    checked = old_bad = 0
    for src in outs:
        (p,) = RC.copy_into(tmp_path, src)
        text = p.read_text(errors="replace")
        m = _DE.search(text)
        if not m:
            p.unlink()
            continue
        idx = oi.build(p)
        e2, de = float(m.group(1)), float(m.group(2))
        en = [b.energy for b in idx.blocks]
        k = en.index(e2)
        prev = [i for i in range(k) if en[i] is not None]
        p.unlink()
        if not prev:          # a RESTART run: its first printed point already has a DE
            continue
        j = prev[-1]
        tol = max(1.5e-3 * abs(de), 3e-11)
        oe = idx.opt_energies()
        assert abs((oe[k] - oe[j]) - de) <= tol, (src.name, oe[k] - oe[j], de)
        old_bad += abs((en[k] - en[j]) - de) > tol
        checked += 1
    assert checked >= 20, checked
    assert old_bad > 0          # the bug this fixes is real in the corpus


@pytest.mark.gui
def test_energy_card_first_step_is_corrected(tmp_path):
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    pytest.importorskip("selenium.webdriver")
    root = H.examples_copy(tmp_path)
    RC.copy_into(root / "runs", RC.DIA_OPT)
    app = U.start_app(tmp_path, root=root)
    try:
        drv = app[0]
        row = drv.find_element("css selector", '#file-list .file[data-path="runs/1_dia_opt_rev1.out"]')
        drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", row)
        row.click()
        H.wait(drv, "return document.querySelectorAll('#o-energy circle.pt').length >= 2", 60)
        titles = drv.execute_script("return [...document.querySelectorAll('#o-energy circle.pt title')].map((t) => t.textContent)")
        vals = [float(re.search(r":\s*(-?[\d.]+(?:e[-+]?\d+)?) meV", t).group(1)) for t in titles]
        assert vals[0] == 0.0
        assert abs((vals[1] - vals[0]) - (-1.24)) < 0.01, vals          # not the -623 meV of the bare SCF energy
        U.no_errors(drv)
    finally:
        U.stop_app(app)
