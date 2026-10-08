"""Compare Convergence (plan E12 fallback) in headless Firefox with real
clicks: Ctrl-click several outputs in the Explorer, right-click › Compare
Convergence opens one plot with a trace per run (energy per atom above the
lowest final energy of the set, each run's stop sentence in its legend).
The shipped demo outputs only (runs in CI)."""
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

RUNS = ["outputs/Ag2Br3_opt_unfinished.out", "outputs/Si_sp.out", "outputs/graphene_slab_sp.out"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    a = U.start_app(tmp_path_factory.mktemp("convergence"))
    try:
        yield a
    finally:
        U.stop_app(a)


def row(drv, path):
    drv.execute_script("window.MACE.state.filter = 'all'; window.MACE.state.text = ''; window.MACE.renderFiles()")
    el = drv.find_element(By.CSS_SELECTOR, f'#file-list .file[data-path="{path}"]')
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    return el


def test_ctrl_click_and_compare_convergence(app):
    drv = app[0]
    U.reset(drv)
    H.wait(drv, "return window.MACE.state.details && window.MACE.state.details['outputs/Si_sp.out']", 60)
    for p in RUNS:
        ActionChains(drv).key_down(Keys.CONTROL).click(row(drv, p)).key_up(Keys.CONTROL).perform()
        time.sleep(0.15)
    picked = drv.execute_script("return [...window.MACE.state.selected]")
    assert sorted(picked) == sorted(RUNS)
    assert not drv.execute_script("return window.__mace.activeDoc()")           # picking opened nothing
    ActionChains(drv).context_click(row(drv, RUNS[0])).perform()
    item = H.wait(drv, "return [...document.querySelectorAll('.menu [role=menuitem]')].find((m) => m.textContent.includes('Compare Convergence'))", 10)
    ActionChains(drv).move_to_element(item).click().perform()
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.kind === 'convergence' && s.drawn", 60)
    fig = drv.execute_script("return window.__mace.plotView.figure()")
    assert len(fig["traces"]) == 3
    names = [t["name"] for t in fig["traces"]]
    assert any(n.startswith("Ag2Br3_opt_unfinished.out · ") for n in names)
    ys = drv.execute_script("return window.__mace.plotView.el().data.map((t) => t.y[t.y.length - 1])")
    assert min(ys) == 0                                                          # the lowest final energy is the reference
    U.shot(drv, "convergence_compare")
    U.no_errors(drv)
