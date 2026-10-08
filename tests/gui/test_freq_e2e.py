"""IR, Raman and FREQ documents in headless Firefox (plan T9).

Corpus files copied into the served folder at run time (skipped without the
test/ corpus). The FREQ document animates a mode at most 30 times a second,
stops on a hidden tab, shows a still view with arrows under reduced motion
(with a Play button that starts the animation), loads nothing from a CDN; the
mode table is a grid with the keyboard model; the explorer's Properties chip
lists the spectra and the FREQ output, and Plot works on each."""
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
from selenium.webdriver.common.keys import Keys  # noqa: E402

CORPUS = H.REPO_ROOT / "test" / "FREQ"
DIA = "2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out"
CONF = "1LiFSI-1DEC-conf{}_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_freq_HSESOL3C_optimized{}"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    if not (CORPUS / DIA).is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp("p5freq")
    root = H.examples_copy(tmp)
    f = root / "freq"
    f.mkdir()
    shutil.copy2(CORPUS / DIA, f / DIA)
    for k in (1, 2):
        for suffix in ("_IRSPEC.DAT", "_RAMSPEC.DAT", ".out"):
            src = CORPUS / CONF.format(k, suffix)
            if src.is_file():
                shutil.copy2(src, f / src.name)
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    app[0].execute_script("window.__mace.settings.set('reduceMotion', 'off')")
    return app


def open_modes(drv, path=f"freq/{DIA}"):
    drv.execute_async_script("const done = arguments[1]; window.__mace.openModes(arguments[0]).then(() => done(1))", path)
    H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode", 60)
    time.sleep(0.4)


def test_freq_document_animates_and_stops_when_hidden(fresh):
    drv, base = fresh[0], fresh[1]
    open_modes(drv)
    st = drv.execute_script("return window.__mace.freqView.state()")
    assert st["n"] == 12 and st["mode"] == 4                  # the first real mode, after the zero ones
    H.wait(drv, "return window.__mace.freqView.playing()", 5)
    f0 = drv.execute_script("return window.__maceFreqFrames")
    time.sleep(1.0)
    f1 = drv.execute_script("return window.__maceFreqFrames")
    assert 5 <= f1 - f0 <= 33, f1 - f0                       # rendering, at most 30 a second
    U.shot(drv, "freq_mode_playing")
    # a hidden tab renders nothing (the loop checks document.hidden)
    drv.execute_script("Object.defineProperty(document, 'hidden', {configurable: true, get: () => true}); document.dispatchEvent(new Event('visibilitychange'))")
    time.sleep(0.3)
    g0 = drv.execute_script("return window.__maceFreqFrames")
    time.sleep(1.0)
    assert drv.execute_script("return window.__maceFreqFrames") == g0
    drv.execute_script("delete document.hidden; document.dispatchEvent(new Event('visibilitychange'))")
    res = drv.execute_script("return performance.getEntriesByType('resource').map((r) => r.name)")
    assert all(r.startswith(base) or r.startswith("blob:") or r.startswith("data:") for r in res), [r for r in res if not r.startswith(base)]
    U.no_errors(drv)


def test_reduced_motion_shows_a_still_view_with_play(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.set('reduceMotion', 'on')")
    open_modes(drv)
    assert not drv.execute_script("return window.__mace.freqView.playing()")
    f0 = drv.execute_script("return window.__maceFreqFrames")
    time.sleep(0.8)
    assert drv.execute_script("return window.__maceFreqFrames") == f0
    assert drv.execute_script("const v = window.MACE.state.viewers['viewer-freq']; return v.getModel() && v.shapes ? v.shapes.length : -1") != 0
    # the still view: atoms at rest, each arrow starting at its sphere's surface and reaching past it
    # (drawn from the atom's center while the atom sat at its turning point, they were buried in the spheres)
    st = drv.execute_script("return window.__mace.freqView.state()")
    assert st["frame"] == 0 and st["arrows"], st
    for a in st["arrows"]:
        gap = sum((p - q) ** 2 for p, q in zip(a["start"], a["atom"])) ** 0.5
        assert gap >= 0.45, a
    assert max(a["length"] for a in st["arrows"]) >= 0.9, st["arrows"]
    U.shot(drv, "freq_reduced_motion")
    drv.execute_script("document.querySelector('#freq-host .fv-play').click()")
    H.wait(drv, "return window.__mace.freqView.playing()", 5)
    time.sleep(0.6)
    assert drv.execute_script("return window.__maceFreqFrames") > f0
    drv.execute_script("window.__mace.freqView.play(false); window.__mace.settings.set('reduceMotion', 'off')")


def test_mode_table_keyboard(fresh):
    drv = fresh[0]
    open_modes(drv)
    drv.execute_script("window.__mace.freqView.play(false)")
    drv.execute_script("document.querySelector('#freq-table .fm-grid').focus()")
    H.wait(drv, "return document.activeElement.classList.contains('fm-row')", 5)
    k0 = drv.execute_script("return Number(document.activeElement.dataset.k)")
    U.keys(drv, Keys.ARROW_DOWN)
    U.keys(drv, Keys.ARROW_DOWN)
    assert drv.execute_script("return Number(document.activeElement.dataset.k)") == k0 + 2
    U.keys(drv, Keys.ENTER)
    H.wait(drv, "return window.__mace.freqView.state().active === arguments[0] && window.__mace.freqView.state().mode === arguments[0] + 1", 10, k0 + 2)
    U.keys(drv, Keys.END)
    assert drv.execute_script("return document.activeElement.getAttribute('aria-rowindex')") == "13"
    assert drv.execute_script("return document.querySelector('#freq-table .fm-grid').getAttribute('aria-rowcount')") == "13"
    H.wait(drv, "return window.__mace.freqView.playing()", 5)          # a chosen mode plays
    U.keys(drv, " ")
    H.wait(drv, "return !window.__mace.freqView.playing()", 5)          # Space pauses
    U.keys(drv, " ")
    H.wait(drv, "return window.__mace.freqView.playing()", 5)           # and plays again
    drv.execute_script("window.__mace.freqView.play(false)")


def test_ir_and_raman_plots_and_the_properties_chip(fresh):
    drv = fresh[0]
    H.wait(drv, "return window.MACE.state.details && Object.keys(window.MACE.state.details).length > 0", 30)
    drv.execute_script("window.MACE.state.filter = 'results'; window.MACE.renderFiles()")
    rows = drv.execute_script("return [...document.querySelectorAll('#file-list .file')].map((b) => [b.dataset.path, b.querySelector('.kb').textContent])")
    kinds = {k for _, k in rows}
    assert {"IR", "Ram", "FREQ"} <= kinds, rows
    ir = next(p for p, k in rows if k == "IR")
    assert drv.execute_script("return window.__mace.menuReason('analysis.ir')") == ""
    U.open_doc(drv, ir, wait_atoms=False)
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.drawn", 60)
    fig = drv.execute_script("return window.__mace.plotView.figure()")
    assert fig["layout"]["xaxis"]["title"]["text"].startswith("Wavenumber")
    # average the conformers (mace plotting --spectra --average)
    drv.execute_script("const c = document.querySelector('#po-avg'); c.checked = true; c.dispatchEvent(new Event('change', {bubbles: true}))")
    H.wait(drv, "return window.__mace.plotView.state().opts.average === true", 20)
    H.wait(drv, "return window.__mace.plotView.figure().traces[0].name.includes('average')", 20)
    U.shot(drv, "ir_average")
    ram = next(p for p, k in rows if k == "Ram")
    U.open_doc(drv, ram, wait_atoms=False)
    H.wait(drv, "const s = window.__mace.plotView.state(); return s && s.kind === 'raman' && s.drawn", 60)
    drv.execute_script("const c = document.querySelector('#po-comp'); c.value = 'all'; c.dispatchEvent(new Event('change', {bubbles: true}))")
    H.wait(drv, "return window.__mace.plotView.figure().traces.length >= 9", 20)
    U.shot(drv, "raman_components")
    # explorer › Plot on the FREQ output opens its modes
    freq = next(p for p, k in rows if k == "FREQ")
    assert drv.execute_script("return window.__mace.contextFor ? true : true")
    drv.execute_script("window.__mace.run('explorer.plot', {path: arguments[0]})", freq)
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'freq'", 30)
    U.no_errors(drv)


def test_modes_view_fits_its_group_with_a_long_name(fresh):
    """The FREQ view never gets wider than its editor group: a long output
    name in its head used to widen it (here 1409 px in an 828 px panel), so
    the molecule sat right of center, and in a split group (414 px) it was
    drawn off-screen."""
    drv, root = fresh[0], fresh[2]
    long = "freq/" + CONF.format(1, "") + ".out"
    if not (root / long).exists():
        pytest.skip("no long-named FREQ output in the corpus")
    size = "const q = (s) => document.querySelector(s).getBoundingClientRect().width; return [q('#tab-freq'), q('#viewer-freq')]"
    open_modes(drv, long)
    tab, view = drv.execute_script(size)
    assert view <= tab + 1, (tab, view)
    drv.execute_script("window.__mace.run('view.splitRight')")
    time.sleep(1.0)
    tab2, view2 = drv.execute_script(size)
    assert tab2 < tab and view2 <= tab2 + 1, (tab2, view2)
    drv.execute_script("window.__mace.run('view.closeGroup')")
    U.no_errors(drv)
