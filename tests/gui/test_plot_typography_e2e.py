"""The labels of every plot the GUI draws (1.1.6), in a real browser: sub- and superscripts in plotly's markup
(E − E<sub>F</sub>, κ<sub>L</sub>, cm<sup>−1</sup>, Å<sup>3</sup>), drawn as such, never a plain-text E_F, cm-1
or ⁻¹ (tests/test_plot_labels.py's scan, here on the figures as plotly holds them), never a "<sub>" shown as
text (the SVG plotly draws), and the plot's table (Copy as TSV, Export CSV) plain.

The kinds of tests/gui/test_plot_legends_all_e2e.py (band, DOS, band + DOS, IR and Raman, transport, Compare
Convergence, a 2D map, a cube slice, Run Analysis) and a phonon fort.25 and ZT with a lattice κ_L. With
MACE_TYPO_SHOTS each figure is saved there (MACE_TYPO_TAG names the set)."""
import json
import os
import shutil
import sys
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
sys.path.insert(0, str(H.REPO_ROOT / "tests"))
import runcorpus as RC  # noqa: E402
import test_plot_legends_all_e2e as LG  # noqa: E402
from test_dos_spin_window import write_doss_spin  # noqa: E402
from test_gui_results import write_band, write_band_crystal  # noqa: E402
from test_parity_fixtures import unpack  # noqa: E402
from test_plot_labels import raw_in, texts  # noqa: E402
from test_plot_legend_outside import write_doss_many  # noqa: E402

SHOTS = os.environ.get("MACE_TYPO_SHOTS")
PH_F25 = H.REPO_ROOT / "tests" / "data" / "f25" / "dia_phonon_f25.fort.25"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("typography")
    root = H.examples_copy(tmp)
    unpack(root / "gp")
    (root / "modes").mkdir()
    shutil.copy(root / "gp" / "water_freq.out", root / "modes" / "water_freq.out")
    (root / "gr").mkdir()
    write_band_crystal(root / "gr", noise=1e-4)
    write_doss_spin(root / "gr")
    (root / "bo").mkdir()
    write_band_crystal(root / "bo", stem="fe_band", beta_shift=0.002)
    (root / "cl").mkdir()
    write_band(root / "cl", stem="mgo_band")
    write_doss_many(root / "cl")
    (root / "ph").mkdir()
    shutil.copy(PH_F25, root / "ph" / PH_F25.name)
    if LG.TRANSPORT.is_dir():
        RC.copy_into(root / "tr", *sorted(LG.TRANSPORT.glob("C1-RCSR-afi_optimized_TRANSPORT.*")))
    if RC.DIA_OPT.is_file():
        RC.copy_into(root / "runs", RC.DIA_OPT, RC.DIA_OPT.with_suffix(".d12"))
    a = U.start_app(tmp, root=root, MACE_GUI_MAX_WORKERS="2")
    a[0].execute_script("window.__mace.settings.set('reduceMotion', 'on')")
    try:
        yield a
    finally:
        U.stop_app(a)


def _tr(patch, cond):
    def go(drv):
        if not LG.TRANSPORT.is_dir():
            pytest.skip("the transport files are in the corpus")
        host = LG._doc("tr/C1-RCSR-afi_optimized_TRANSPORT.SEEBECK.dat")(drv)
        drv.execute_script("window.__mace.plotView.setOpt(arguments[0], 'test')", patch)
        H.wait(drv, "const s = window.__mace.plotView.state(); return !!(s && s.drawn && s.transport && " + cond + ")", 60)
        return host
    return go


_zt = _tr({"quantity": "zt", "kl": "1.5"}, "s.transport.quantity === 'zt' && !s.transport.upper_bound")
_pf_carriers = _tr({"quantity": "pf", "x": "carriers"}, "s.transport.quantity === 'pf'")


def _shifted(drv):
    host = LG._doc("gr/gra_band.BAND.DAT")(drv)
    drv.execute_script("window.__mace.plotView.setOpt({shift: 0.5}, 'test')")
    H.wait(drv, "const el = window.__mace.plotView.el(); return el.layout && /0.50 eV/.test(el.layout.yaxis.title.text)", 30)
    return host


CASES = {**LG.CASES, "phonon f25": LG._doc("ph/dia_phonon_f25.fort.25"), "transport zt kappa_L": _zt,
         "transport pf carriers": _pf_carriers, "band+dos shifted": _shifted}

# the figure as plotly holds it, without its numbers
FIGURE = """const el = document.querySelector(arguments[0]);
const drop = (k, v) => (ArrayBuffer.isView(v) || (Array.isArray(v) && v.length && typeof v[0] !== 'string' && !(v[0] && typeof v[0] === 'object')) ? undefined : v);
return el && el.data ? JSON.stringify({data: el.data, layout: el.layout}, drop) : null;"""
# a tag or an escaped one shown as text in the SVG plotly drew
SHOWN = """const el = document.querySelector(arguments[0]);
return [...el.querySelectorAll('text, tspan')].map((t) => t.textContent).filter((s) => /<\\/?(sub|sup|br)>|&lt;|&gt;/.test(s));"""


@pytest.mark.parametrize("case", list(CASES))
def test_plot_labels_are_typeset(app, case):
    drv = app[0]
    U.reset(drv)
    drv.execute_script("window.__mace.closeAll()")
    host, _ = CASES[case](drv)
    time.sleep(0.6)
    if SHOTS:
        os.makedirs(SHOTS, exist_ok=True)
        tag = os.environ.get("MACE_TYPO_TAG", "after")
        drv.save_screenshot(os.path.join(SHOTS, f"{tag}_{case.replace(' ', '_').replace('+', '_')}.png"))
    fig = json.loads(drv.execute_script(FIGURE, host))
    bad = [(w, t, raw_in(t)) for w, t in texts(fig) if raw_in(t)]
    assert not bad, bad[:8]
    assert not drv.execute_script(SHOWN, host)
    # nor anywhere on the page: the subtitle, the plot options, the readout
    shown = drv.execute_script("return document.body.innerText.split('\\n').filter((s) => /<\\/?(sub|sup)>/.test(s))")
    assert not shown, shown
    # the plot's table (Copy as TSV, Export CSV): plain text, ASCII headers
    view = "runView" if host.startswith("#runplot") else "plotView" if host.startswith("#plot") else None
    if view:
        tab = drv.execute_script(f"const v = window.__mace.{view}; return v && v.table ? v.table() : null")
        if tab:
            heads = [str(c) for c in tab["columns"]]
            assert not [c for c in heads if "<" in c or ">" in c], heads
            if case.startswith("transport"):
                assert all(c.isascii() for c in heads), heads
    U.no_errors(drv)
