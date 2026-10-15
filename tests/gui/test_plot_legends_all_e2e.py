"""Every plot the GUI draws keeps its legends, colorbars and annotation boxes off
the data, the tick labels, the axis titles, each other, the subtitle and the
figure edge, at UI scale 100/110/125 % and window widths 900/1366/1920, in
Firefox and Chromium (tests/legend_geometry.py: the rule the CLI's figures are
held to as well).

The kinds: band + DOS (spin-polarized), band only, DOS only, a DOS with ten
projections; IR and Raman from CRYSTAL's SPEC.DAT with the mode markers and
the peaks labeled, Raman with every component, IR broadened from the modes;
BOLTZTRA transport (σ, S, κ, every component); Compare Convergence; a 2D
charge map with its colorbar; a cube slice with its colorbar; Run Analysis
(energy, convergence, SCF, cell). A new plot kind is added to CASES. The
transport files and the OPT run come from the corpus (skipped without it).
"""
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
from legend_geometry import PLOTLY_BOXES, plotly_problems  # noqa: E402
from test_dos_spin_window import write_doss_spin  # noqa: E402
from test_gui_results import write_band, write_band_crystal  # noqa: E402
from test_parity_fixtures import unpack  # noqa: E402
from test_plot_legend_outside import write_doss_many  # noqa: E402

CORPUS = H.REPO_ROOT / "test"
TRANSPORT = CORPUS / "TRANSPORT"
SHOTS = os.environ.get("MACE_LEGEND_SHOTS")


@pytest.fixture(scope="module", params=["firefox", "chromium"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("legends")
    root = H.examples_copy(tmp)
    unpack(root / "gp")
    (root / "modes").mkdir()
    shutil.copy(root / "gp" / "water_freq.out", root / "modes" / "water_freq.out")      # no SPEC.DAT: from the modes
    (root / "gr").mkdir()
    write_band_crystal(root / "gr", noise=1e-4)
    write_doss_spin(root / "gr")
    (root / "bo").mkdir()
    write_band_crystal(root / "bo", stem="fe_band", beta_shift=0.002)
    (root / "cl").mkdir()
    write_band(root / "cl", stem="mgo_band")
    write_doss_many(root / "cl")
    if TRANSPORT.is_dir():
        RC.copy_into(root / "tr", *sorted(TRANSPORT.glob("C1-RCSR-afi_optimized_TRANSPORT.*")))
    if RC.DIA_OPT.is_file():
        RC.copy_into(root / "runs", RC.DIA_OPT, RC.DIA_OPT.with_suffix(".d12"))
    real = H.firefox
    if request.param == "chromium":
        H.firefox = lambda w, h, download_dir=None, **kw: H.chromium(w, h, download_dir=download_dir)
    try:
        a = U.start_app(tmp, root=root, MACE_GUI_MAX_WORKERS="2")
    finally:
        H.firefox = real
    a[0].execute_script("window.__mace.settings.set('reduceMotion', 'on')")
    try:
        yield a + (request.param,)
    finally:
        try:
            a[0].execute_script("window.__mace.settings.set('uiScale', 100)")
        except Exception:
            pass
        U.stop_app(a)


PLOT = ("#plot-host .pv-plot", "#plot-host .pv-note")
RUN = ("#runplot-host .rp-plot", "#runplot-host .pv-note")
SLICE = ("#volume-host .vv-slice-plot", None)


def _doc(path, opts=()):
    def go(drv):
        U.open_doc(drv, path, wait_atoms=False)
        H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return !!(s && s.drawn && v.el().dataset.drawn === arguments[0])", 60, path)
        for ident, value in opts:
            drv.execute_script("const i = document.getElementById(arguments[0]); if (i.type === 'checkbox') i.checked = !!arguments[1]; else i.value = arguments[1];"
                               " i.dispatchEvent(new Event('change', {bubbles: true}))", ident, value)
            time.sleep(0.8)
        return PLOT
    return go


def _run(tab):
    def go(drv):
        if not RC.DIA_OPT.is_file():
            pytest.skip("the OPT run is in the corpus")
        drv.execute_script("window.__maceOpenRunPlot(arguments[0], arguments[1])", "runs/1_dia_opt_rev1.out", tab)
        H.wait(drv, "const v = window.__mace.runView, s = v && v.state(); return !!(s && s.drawn && s.drawn.endsWith('|' + arguments[0]))", 60, tab)
        return RUN
    return go


def _modes(drv):
    drv.execute_script("window.__mace.openModes('modes/water_freq.out')")
    H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return !!(s && s.mode)", 60)
    drv.execute_script("window.__mace.freqView.play(false); document.querySelector('#freq-host [data-act=ir]').click()")
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return !!(s && s.drawn && s.source === 'modes')", 60)
    return PLOT


def _convergence(drv):
    drv.execute_script("window.__mace.openConvergence(arguments[0])",
                       ["outputs/Ag2Br3_opt_unfinished.out", "outputs/Si_sp.out", "outputs/graphene_slab_sp.out"])
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return !!(s && s.kind === 'convergence' && s.drawn)", 60)
    return PLOT


def _slice(drv):
    U.open_doc(drv, "gp/mgo_dens/DENS_CUBE.DAT", wait_atoms=False)
    H.wait(drv, "const v = window.__mace.volumeView, s = v && v.state(); return !!(s && s.type)", 60)
    drv.execute_script("document.getElementById('vo-slice').click()")
    H.wait(drv, "const p = document.querySelector('#volume-host .vv-slice-plot'); return !!(p && p.data && p.data.length)", 60)
    return SLICE


def _transport(name, comp="all"):
    def go(drv):
        if not TRANSPORT.is_dir():
            pytest.skip("the transport files are in the corpus")
        return _doc(f"tr/C1-RCSR-afi_optimized_TRANSPORT.{name}.dat", [("po-tcomp", comp)])(drv)
    return go


CASES = {
    "band+dos spin": _doc("gr/gra_band.BAND.DAT"),
    "band only": _doc("bo/fe_band.BAND.DAT"),
    "dos only": _doc("gr/gra_doss.DOSS.DAT"),
    "band+dos many": _doc("cl/mgo_band.BAND.DAT"),
    "ir + modes": _doc("gp/water_freq_IRSPEC.DAT"),
    "raman all components": _doc("gp/water_freq_RAMSPEC.DAT", [("po-comp", "all")]),
    "ir from modes": _modes,
    "transport sigma": _transport("SIGMA"),
    "transport seebeck": _transport("SEEBECK"),
    "transport kappa": _transport("KAPPA"),
    "compare convergence": _convergence,
    "2d map": _doc("gp/mgo_echg.f25"),
    "cube slice": _slice,
    "run energy": _run("energy"),
    "run convergence": _run("convergence"),
    "run scf": _run("scf"),
    "run cell": _run("cell"),
}


def _settle(drv):
    time.sleep(0.4)
    drv.execute_script("window.dispatchEvent(new Event('resize'))")
    time.sleep(0.9)


@pytest.mark.parametrize("case", list(CASES))
def test_plot_boxes_stay_off_the_data(app, case):
    drv, browser = app[0], app[-1]
    U.reset(drv)
    drv.execute_script("window.__mace.closeAll()")
    host, note = CASES[case](drv)
    time.sleep(0.5)
    failures = []
    for scale in (100, 110, 125):
        drv.execute_script("window.__mace.settings.set('uiScale', arguments[0])", scale)
        for width in (900, 1366, 1920):
            H.set_viewport(drv, width, 900)
            _settle(drv)
            st = drv.execute_script(PLOTLY_BOXES, host, note) if note else drv.execute_script(PLOTLY_BOXES, host)
            failures += [f"{scale}% {width}px: {p}" for p in plotly_problems(st)]
            if note and width == 1366 and st and st["note"] and st["note"]["lines"] > 1 and st["note"]["text"].strip():
                failures.append(f"{scale}% {width}px: the subtitle takes {st['note']['lines']} lines")
            if SHOTS and scale == 110 and width == 1366:
                os.makedirs(SHOTS, exist_ok=True)
                tag = os.environ.get("MACE_LEGEND_TAG", "after")
                drv.save_screenshot(os.path.join(SHOTS, f"{tag}_{case.replace(' ', '_').replace('+', '_')}_{browser}_110.png"))
    drv.execute_script("window.__mace.settings.set('uiScale', 100)")
    H.set_viewport(drv, 1280, 800)
    assert not failures, "\n".join(dict.fromkeys(failures))
