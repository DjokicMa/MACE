"""The OPT compare view's Displacement map and arrows on real optimizations, in
a real browser (MACE_GUI_BROWSER=firefox, the default, or chromium).

Ag1Br1 (corpus): only the cell relaxes, the atoms sit on special positions, so
the displacement relative to the cell is zero (the maintainer's AB graphite:
no arrows, a 0.0000 Å to 0.0000 Å scale). Δr: With cell change (the default
when the cell changed) draws the arrows; Relative to the cell says why
nothing moves and how the cell changed. Ag2Cl1: the atoms relax too."""
import os
import re
import time

import pytest

import guiharness as H
import runcorpus as RC

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

FIXED = RC.CORPUS / "OPT" / "Ag1Br1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.out"
MOVED = RC.CORPUS / "OPT" / "Ag2Cl1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.out"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    RC.need(FIXED)
    RC.need(MOVED)
    tmp = tmp_path_factory.mktemp("dmap")
    root = H.examples_copy(tmp)
    RC.copy_into(root / "runs", FIXED, MOVED, *sorted((RC.CORPUS / "OPT").glob("LiFSI_MOLECULE_OPT*.out"))[:1])
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = None
    try:
        drv = H.browser(os.environ.get("MACE_GUI_BROWSER", "firefox"), width=1400, height=900)
        drv.set_script_timeout(120)
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'on')")
        yield drv
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def open_run(drv, src):
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())", f"runs/{src.name}")
    H.wait(drv, "return !!(window.MACE.state.traj && window.MACE.state.cmp && window.MACE.state.viewers['viewer-before'])", 60)
    drv.execute_script("window.MACE.setLook('dmap')")
    H.wait(drv, "return document.querySelector('#o-legend').dataset.look === 'dmap'", 10)
    time.sleep(0.5)


def view(drv):
    return drv.execute_script("""
        const v = window.MACE.state.viewers['viewer-before'], lg = document.querySelector('#o-legend');
        const w = document.querySelector('#o-dmode-wrap');
        return { arrows: v.arrowCount || 0, legend: lg.textContent, bar: !!lg.querySelector('.seq-bar'),
          note: (lg.querySelector('.lg-note') || {}).textContent || '', mode: document.querySelector('#o-dmode').value,
          modeShown: !w.hidden && w.offsetParent !== null, scale: Number(document.querySelector('#o-scale').value),
          stats: document.querySelector('#o-stats').textContent }""")


def pick_mode(drv, mode):
    from selenium.webdriver.support.ui import Select
    Select(drv.find_element("css selector", "#o-dmode")).select_by_value(mode)
    H.wait(drv, "return document.querySelector('#o-dmode').value === arguments[0]", 10, mode)
    time.sleep(0.3)


def degenerate(text):
    return "0.0000 Å" in text or re.search(r"\b0 Å\D+0 Å", text) is not None


def test_cell_only_relaxation_draws_arrows_and_explains_internal(app):
    drv = app
    drv.execute_script("localStorage.removeItem('mace.gui.cmp.dispMode')")
    open_run(drv, FIXED)
    v = view(drv)
    # the cell changed: Δr is offered and defaults to With cell change, which moves the atoms
    assert v["modeShown"] and v["mode"] == "cart"
    assert v["arrows"] > 0, v
    assert not degenerate(v["legend"]), v["legend"]
    assert "1 Å drawn = " in v["legend"] and f"×{v['scale']}" in v["legend"]
    assert "Cell: a" in v["legend"]
    assert v["scale"] > 1                                           # picked from the largest displacement
    assert "0.000" not in v["stats"]
    # Relative to the cell: nothing moves; say so, with the cell's change, and draw no arrows
    pick_mode(drv, "internal")
    v = view(drv)
    assert v["arrows"] == 0
    assert v["note"].startswith("Atoms did not move relative to the cell"), v["note"]
    assert "the cell changed: a" in v["note"] and "%" in v["note"]
    assert not v["bar"] and not degenerate(v["legend"])
    # remembered: the next run opens on the same choice
    open_run(drv, FIXED)
    assert view(drv)["mode"] == "internal"
    pick_mode(drv, "cart")
    H.wait(drv, "return window.MACE.state.viewers['viewer-before'].arrowCount > 0", 10)
    assert not H.errors(drv)


def test_real_relaxation_in_both_modes(app):
    drv = app
    open_run(drv, MOVED)
    for mode in ("internal", "cart"):
        pick_mode(drv, mode)
        v = view(drv)
        assert v["arrows"] > 0, (mode, v)
        assert v["bar"] and not degenerate(v["legend"]), v["legend"]
        ends = re.findall(r"(\d[\d.e+-]*) (Å|pm)", v["legend"].split("|Δr|")[1])[:2]
        assert len(ends) == 2 and ends[0] != ends[1], v["legend"]
    # the other looks carry the arrow scale too
    for look in ("overlay", "side"):
        drv.execute_script("window.MACE.setLook(arguments[0])", look)
        H.wait(drv, "return document.querySelector('#o-legend').dataset.look === arguments[0]", 10, look)
        assert "1 Å drawn = " in view(drv)["legend"]
        side = drv.execute_script("const v = window.MACE.state.viewers[arguments[0]]; return v ? v.arrowCount : 0",
                                  "viewer-after" if look == "side" else "viewer-before")
        assert side > 0
    drv.execute_script("window.MACE.setLook('overlay')")
    assert not H.errors(drv)


def test_molecule_offers_no_cell_choice(app):
    drv = app
    mol = sorted((RC.CORPUS / "OPT").glob("LiFSI_MOLECULE_OPT*.out"))
    if not mol:
        pytest.skip("no molecule OPT in the corpus")
    open_run(drv, mol[0])
    v = view(drv)
    assert not v["modeShown"]                                        # no Δr choice for a molecule
    assert "relative to the cell" not in v["legend"] and not degenerate(v["legend"]), v["legend"]
    assert not H.errors(drv)
