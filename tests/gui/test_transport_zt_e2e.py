"""The transport plot's ZT view, in a real browser (1.1.6): the ZT button
beside the quantities, a lattice κ_L (a value, then a T,kappa_L CSV in the
folder) and τ typed into Plot options, the ZTₑ and ZT total curves, and the
plotted numbers equal to what `mace analyze --extract-properties
--kappa-lattice` stores for the same run (the extractor in this process).
The run is a corpus TRANSPORT run (test/, read only, copied; skipped without
it). The browser zoom is MACE_GUI_ZOOM (default 1.0); with MACE_GUI_SHOTS
the figure is saved there."""
import math
import os
import shutil
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

TRANSPORT = H.REPO_ROOT / "test" / "TRANSPORT"
STEM = "C1-RCSR-bcu-f_optimized_TRANSPORT"
ZOOM = float(os.environ.get("MACE_GUI_ZOOM", "1.0"))


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    if not TRANSPORT.is_dir():
        pytest.skip("test/ corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp("transport_zt")
    root = tmp / "root"
    root.mkdir()
    for q in (".out", ".SEEBECK.dat", ".SIGMA.dat", ".KAPPA.dat"):
        shutil.copyfile(TRANSPORT / (STEM + q), root / ("run" + q))
    (root / "kl.csv").write_text("T,kappa_L\n300,3.0\n500,2.0\n700,1.0\n")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2")
    w, h = round(1366 / ZOOM), round(820 / ZOOM)
    drv = None
    try:
        drv = H.firefox(1366, 820, prefs={"layout.css.devPixelsPerPx": str(ZOOM)} if ZOOM != 1.0 else None)
        H.set_viewport(drv, w, h)
        H.load(drv, base)
        yield drv, root
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def click(drv, el):
    drv.execute_script("arguments[0].scrollIntoView({block: 'nearest', inline: 'nearest'})", el)
    ActionChains(drv).move_to_element(el).pause(0.1).click().perform()


def type_into(drv, css, text):
    """Real keys into a Plot options field; the options are drawn again after
    every change, so the field is looked up again when a redraw replaced it."""
    from selenium.common.exceptions import StaleElementReferenceException
    for _ in range(5):
        try:
            el = drv.find_element(By.CSS_SELECTOR, css)
            click(drv, el)
            el.send_keys(Keys.CONTROL, "a")
            el.send_keys(text)
            el.send_keys(Keys.TAB)            # a change event
            return
        except StaleElementReferenceException:
            time.sleep(0.3)
    raise AssertionError(f"{css} kept being replaced")


def state(drv):
    return drv.execute_script("return window.__mace.plotView.state()")


def traces(drv):
    return drv.execute_script("return window.__mace.plotView.el().data.map((t) => ({meta: t.meta, name: t.name, x: Array.from(t.x), y: Array.from(t.y)}))")


def cli_props(root, kappa_l, tau=None):
    from mace.utils import transport_zt as Z
    from mace.utils.property_extractor import CrystalPropertyExtractor
    x = CrystalPropertyExtractor(enable_tracking=False)
    x.kappa_lattice = Z.parse_kappa_lattice(kappa_l)
    x.tau_fs = tau
    return x.extract_all_properties(root / "run.out")


def test_zt_view_shows_the_clis_numbers(app):
    drv, root = app
    drv.execute_script(U.RESET_JS)
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())", "run.SEEBECK.dat")
    H.wait(drv, "const v = window.__mace.plotView; const s = v && v.state(); return s && s.kind === 'transport' && s.drawn", 60)
    click(drv, drv.find_element(By.CSS_SELECTOR, ".po-tq [data-tview='zt']"))
    H.wait(drv, "const v = window.__mace.plotView; const s = v && v.state(); return s.drawn && s.zt && s.zt.zt_e_max", 60)
    assert "Figure of merit ZT" == drv.execute_script("return window.__mace.plotView.el().layout.yaxis.title.text")
    assert [t["meta"] for t in traces(drv)] == ["zt_e"]
    assert drv.find_element(By.CSS_SELECTOR, ".po-tq [data-tview='zt']").get_attribute("aria-checked") == "true"

    # a value: the CLI's peak, to the last digit
    type_into(drv, "#po-kl", "1.5")
    H.wait(drv, "const v = window.__mace.plotView; const s = v && v.state(); return s.drawn && s.zt && s.zt.zt_total_max", 60)
    p = cli_props(root, "1.5")
    s = state(drv)["zt"]
    assert s["zt_total_max"]["value"] == p["transport_zt_total_max_avg"]
    assert s["zt_e_max"]["value"] == p["transport_zt_electronic_max_avg"]
    assert s["tau_fs"] == p["transport_zt_total_tau_fs"] == 22.0
    assert s["kappa_l_source"] == p["transport_zt_total_lattice_source"]
    # the plotted curve at the peak's temperature reaches that value at its μ
    T = p["transport_zt_total_max_temperature_k"]
    drv.execute_script("window.__mace.plotView.setOpt({T: arguments[0]}, 'T')", T)
    H.wait(drv, "const v = window.__mace.plotView; return v && v.state() && v.state().drawn", 30)
    tr = {t["meta"]: t for t in traces(drv)}
    assert set(tr) == {"zt_e", "zt_total"}
    assert "electrons only" in tr["zt_e"]["name"] and "κ_L = 1.5" in tr["zt_total"]["name"] and "τ 22 fs" in tr["zt_total"]["name"]
    ys = [y for y in tr["zt_total"]["y"] if y is not None]
    assert max(ys) == p["transport_zt_total_max_avg"]
    i = tr["zt_total"]["x"].index(p["transport_zt_total_max_mu_ev"])
    assert tr["zt_total"]["y"][i] == p["transport_zt_total_max_avg"]
    # the total sits below ZTₑ wherever both are drawn
    assert all(t < e for t, e in zip(tr["zt_total"]["y"], tr["zt_e"]["y"]) if t is not None and e is not None)

    # a table in the folder and a tau: the same again
    type_into(drv, "#po-kl", "kl.csv")
    type_into(drv, "#po-tau", "44")
    H.wait(drv, "const v = window.__mace.plotView; const s = v && v.state(); return s.drawn && s.zt && s.zt.tau_fs === 44 && s.zt.zt_total_max", 60)
    p = cli_props(root, str(root / "kl.csv"), 44.0)
    s = state(drv)["zt"]
    assert s["zt_total_max"]["value"] == p["transport_zt_total_max_avg"]
    assert (s["zt_total_max"]["temperature_k"], s["zt_total_max"]["mu_ev"]) == (
        p["transport_zt_total_max_temperature_k"], p["transport_zt_total_max_mu_ev"])
    notes = drv.execute_script("return [...document.querySelectorAll('.po-note')].map((e) => e.textContent).join(' | ')")
    assert "ZT total peak:" in notes and "τ 44 fs" in notes and "the run used 22 fs" in notes
    drv.execute_script("window.__mace.plotView.setOpt({T: arguments[0]}, 'T')", p["transport_zt_total_max_temperature_k"])
    H.wait(drv, "const v = window.__mace.plotView; return v && v.state() && v.state().drawn", 30)
    time.sleep(0.5)
    U.shot(drv, f"transport_zt_{round(ZOOM * 100)}pct")
    errs = [e for e in H.errors(drv) if "ResizeObserver loop" not in e]
    assert not errs, errs


def test_refusals_are_shown(app):
    drv, root = app
    (root / "short.csv").write_text("300,3.0\n500,2.0\n")
    drv.execute_script(U.RESET_JS)
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())", "run.SIGMA.dat")
    H.wait(drv, "const v = window.__mace.plotView; const s = v && v.state(); return s && s.kind === 'transport' && s.drawn", 60)
    click(drv, drv.find_element(By.CSS_SELECTOR, ".po-tq [data-tview='zt']"))
    H.wait(drv, "const v = window.__mace.plotView; const s = v && v.state(); return s.drawn && s.zt", 60)
    # a table that stops short of the run's 700 K: no peak, the reason, and no curve at 700 K
    type_into(drv, "#po-kl", "short.csv")
    H.wait(drv, "const v = window.__mace.plotView; const s = v && v.state(); return s.drawn && s.zt && s.zt.refused", 60)
    assert "700 K" in state(drv)["zt"]["refused"]
    drv.execute_script("window.__mace.plotView.setOpt({T: 700}, 'T')")
    H.wait(drv, "const v = window.__mace.plotView; return v && v.state() && v.state().drawn", 30)
    assert [t["meta"] for t in traces(drv)] == ["zt_e"]
    assert "700 K" in drv.find_element(By.CSS_SELECTOR, ".po-note.po-err").text
    # a value MACE refuses: the message, and ZTₑ alone
    type_into(drv, "#po-kl", "-1")
    H.wait(drv, "const v = window.__mace.plotView; const s = v && v.state(); return s.drawn && s.zt && s.zt.error", 60)
    assert "positive" in state(drv)["zt"]["error"]
    assert [t["meta"] for t in traces(drv)] == ["zt_e"]
    # back to the file's own quantity
    click(drv, drv.find_element(By.CSS_SELECTOR, ".po-tq [data-tq='run.SIGMA.dat']"))
    H.wait(drv, "const v = window.__mace.plotView; const s = v && v.state(); return s.drawn && s.opts.view === 'quantity'", 30)
    assert "conductivity" in drv.execute_script("return window.__mace.plotView.el().layout.yaxis.title.text")
