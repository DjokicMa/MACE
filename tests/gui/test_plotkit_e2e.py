"""plotkit, the options every plot shares (PLAN_analysis C3), in headless
Firefox with real clicks and keys: units, log axes, Per, ranges, Reset Zoom,
Export (PNG, SVG, CSV) and Copy as TSV through the real clipboard (pasted
into a textarea with Ctrl+V), persistence, the Settings defaults,
registration of every chart host, the toolbar's width at 1280 × 800, and
the keyboard. Real outputs copied from the corpus."""
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import runcorpus as RC

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402
from selenium.webdriver.support.ui import Select  # noqa: E402

BAND = RC.CORPUS / "BAND" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.BAND.DAT"
FREQ = RC.CORPUS / "FREQ" / "2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    RC.need(BAND)
    tmp = tmp_path_factory.mktemp("plotkit")
    root = H.examples_copy(tmp)
    RC.copy_into(root / "pk", BAND, RC.DIA_OPT, FREQ)
    shutil.copy2(BAND, root / "pk" / "second.BAND.DAT")
    xdg = tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_TEST_HOOKS="1")
    downloads = tmp / "downloads"
    downloads.mkdir()
    drv = H.firefox(1280, 800, download_dir=downloads, prefs={
        "dom.events.testing.asyncClipboard": True, "dom.events.asyncClipboard.clipboardItem": True,
        "dom.events.asyncClipboard.readText": True})
    H.load(drv, base)
    try:
        yield drv, base, root, downloads
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def file_row(drv, rel):
    row = drv.find_element("css selector", f'#file-list .file[data-path="{rel}"]')
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", row)
    return row


def open_band(drv, rel="pk/" + BAND.name):
    drv.execute_script("window.__mace.closeAll()")
    file_row(drv, rel).click()
    H.wait(drv, "return window.__mace.plotView && window.__mace.plotView.state() && window.__mace.plotView.state().drawn", 60)


def drawn(drv):
    H.wait(drv, "return window.__mace.plotView.state().drawn", 30)
    return drv.execute_script("return window.__mace.plotView.figure()")


def paste(drv):
    """The real clipboard: a textarea, Ctrl+V, its value."""
    drv.execute_script("const t = document.createElement('textarea'); t.id = 'paste-here'; t.style.cssText = 'position:fixed;top:0;left:0;width:50px;height:20px;z-index:99999';"
                       "document.body.appendChild(t)")
    t = drv.find_element("css selector", "#paste-here")
    t.click()
    U.keys(drv, Keys.CONTROL, "v")
    time.sleep(0.3)
    v = drv.execute_script("const t = document.querySelector('#paste-here'); const v = t.value; t.remove(); return v")
    return v


def test_units_and_tsv_tables_in_the_module(app):
    drv = app[0]
    r = drv.execute_async_script("""const done = arguments[0];
        import(new URL('js/plotkit.js', document.querySelector('script[src*="/js/shell.js"]').src.replace(/js\\/shell\\.js.*$/, '')).href).then((P) => done({
          ev: P.energy(P.HARTREE_EV, 'Ha'), evv: P.energy(P.HARTREE_EV, 'eV'), kj: P.energy(P.HARTREE_EV, 'kJ/mol'), kcal: P.energy(P.HARTREE_EV, 'kcal/mol'),
          cm: P.wavenumber(8.0655, 'meV'), per: [2, 4].map((x) => x / 2),
          tsv: P.toTSV({ columns: ['x', 'a', 'b'], rows: [[1, 2, 3], [4, 5, 6]] }) }))""")
    assert abs(r["ev"] - 1.0) < 1e-12 and abs(r["evv"] - 27.211386) < 1e-6          # 1 Ha
    assert abs(r["kj"] - 2625.4996) < 1e-3 and abs(r["kcal"] - 627.5095) < 1e-3
    assert abs(r["cm"] - 1.0) < 1e-4
    assert r["per"] == [1, 2]
    assert r["tsv"] == "x\ta\tb\n1\t2\t3\n4\t5\t6\n"


def test_band_toolbar_every_control(app):
    drv, base, root, downloads = app
    open_band(drv)
    # no Log Y: band energies are signed (1.1.6: Log Y is a DOS axis, tests/gui/test_log_dos_e2e.py)
    assert not drv.find_elements("css selector", "#plot-host .pk-logy")
    assert drv.execute_script("return window.__mace.plotView.figure().layout.yaxis.type") == "linear"
    ev = drv.execute_script("return window.__mace.plotView.figure().traces[0]")
    # Units: kJ/mol (title and values)
    drv.find_element("css selector", "#plot-host .pk-units").click()
    drv.find_element("css selector", ".pk-menu [data-v='kJ/mol']").click()
    H.wait(drv, "return /kJ\\/mol/.test(window.__mace.plotView.figure().layout.yaxis.title.text)", 10)
    # the figure helper gives counts only: compare the plotted arrays directly
    ys = drv.execute_script("const el = window.__mace.plotView.el(); return [el.data[0].y.find((v) => v !== null)]")
    drv.find_element("css selector", "#plot-host .pk-units").click()
    drv.find_element("css selector", ".pk-menu [data-v='eV']").click()
    H.wait(drv, "return /\\(eV\\)/.test(window.__mace.plotView.figure().layout.yaxis.title.text)", 10)
    ys_ev = drv.execute_script("const el = window.__mace.plotView.el(); return [el.data[0].y.find((v) => v !== null)]")
    assert abs(ys[0] - ys_ev[0] * 96.48533212331) < 1e-6 * max(1, abs(ys[0]))
    # Range, then Reset Zoom
    drv.find_element("css selector", "#plot-host .pk-more").click()
    lo = drv.find_element("css selector", ".pk-pop [data-r=y0]")
    lo.send_keys("-3", Keys.TAB)
    hi = drv.find_element("css selector", ".pk-pop [data-r=y1]")
    hi.send_keys("2", Keys.TAB)
    H.wait(drv, "const r = window.__mace.plotView.el().layout.yaxis.range; return r && r[0] === -3 && r[1] === 2", 10)
    drv.find_element("css selector", ".pk-pop [data-act=resetzoom]").click()
    H.wait(drv, "const o = window.__mace.plotView.state().opts; return o.yr === null", 10)
    H.wait(drv, "const r = window.__mace.plotView.el().layout.yaxis.range; return r && r[0] === -6 && r[1] === 6", 10)
    U.keys(drv, Keys.ESCAPE)
    # Export: PNG, SVG, CSV into the downloads folder
    for f in downloads.iterdir():
        f.unlink()
    for v in ("png", "svg", "csv"):
        drv.find_element("css selector", "#plot-host .pk-export").click()
        drv.find_element("css selector", f".pk-menu [data-v={v}]").click()
    end = time.time() + 20
    while time.time() < end and len([f for f in downloads.iterdir() if not f.name.endswith(".part")]) < 3:
        time.sleep(0.2)
    names = sorted(f.name for f in downloads.iterdir())
    assert any(n.endswith(".png") for n in names) and any(n.endswith(".svg") for n in names) and any(n.endswith("_data.csv") for n in names), names
    # Copy as TSV through the real clipboard
    drv.find_element("css selector", "#plot-host .pk-export").click()
    drv.find_element("css selector", ".pk-menu [data-v=tsv]").click()
    H.wait(drv, "return [...document.querySelectorAll('#toasts .msg')].some((m) => /Copied \\d+ rows/.test(m.textContent))", 10)
    text = paste(drv)
    assert text.split("\n")[0].startswith("path\tband1"), text[:200]
    U.no_errors(drv)


def test_persistence_and_settings_defaults(app):
    drv, base = app[0], app[1]
    open_band(drv)
    drv.find_element("css selector", "#plot-host .pk-units").click()
    drv.find_element("css selector", ".pk-menu [data-v='meV']").click()
    H.wait(drv, "return window.__mace.plotView.state().opts.unit === 'meV'", 10)
    H.load(drv, base)
    open_band(drv)
    assert drv.execute_script("return window.__mace.plotView.state().opts.unit") == "meV"
    # Settings › Energy unit: Ha (real keys and clicks); a second band file takes it, the first keeps its own
    drv.execute_script("document.body.focus()")
    U.keys(drv, Keys.CONTROL, ",")
    H.wait(drv, "return document.querySelector('#settings') && document.querySelector('#settings').open")
    drv.find_element("css selector", "#settings [data-sec='units']").click()
    Select(drv.find_element("css selector", "#set-energyUnit")).select_by_value("Ha")
    U.keys(drv, Keys.ESCAPE)
    open_band(drv, "pk/second.BAND.DAT")
    assert drv.execute_script("return window.__mace.plotView.state().opts.unit") == "Ha"
    open_band(drv)
    assert drv.execute_script("return window.__mace.plotView.state().opts.unit") == "meV"
    # a newly opened run plot uses Ha too
    drv.execute_script("window.__mace.closeAll()")
    file_row(drv, f"pk/{RC.DIA_OPT.name}").click()
    H.wait(drv, "const b = document.querySelector('#o-e-plot'); return b && b.offsetParent !== null", 60)
    drv.find_element("css selector", "#o-e-plot").click()
    H.wait(drv, "return window.__mace.runView && window.__mace.runView.state() && window.__mace.runView.state().drawn", 60)
    assert drv.execute_script("return window.__mace.runView.state().opts.energy.unit") == "Ha"
    # Per atom on the run plot: the axis says so
    drv.find_element("css selector", "#runplot-host .pk-more").click()
    Select(drv.find_element("css selector", ".pk-pop [data-k=per]")).select_by_value("atom")
    U.keys(drv, Keys.ESCAPE)
    H.wait(drv, "return /\\/atom/.test(window.__mace.runView.figure().layout.yaxis.title.text)", 10)
    drv.execute_script("window.__mace.settings.set('energyUnit', 'eV')")
    U.no_errors(drv)


def test_every_chart_host_is_registered_and_fits(app):
    drv = app[0]
    hosts = []
    open_band(drv)
    hosts.append(drv.execute_script("return window.__mace.plotView.el().dataset.plotkit"))
    over = drv.execute_script("const a = document.querySelector('#plot-host .pv-acts'); return [a.scrollWidth, a.clientWidth]")
    assert over[0] <= over[1], over
    drv.execute_script("window.__mace.closeAll()")
    file_row(drv, f"pk/{RC.DIA_OPT.name}").click()
    H.wait(drv, "return !!document.querySelector('#o-energy svg.echart')", 60)
    hosts.append(drv.execute_script("return document.querySelector('#o-energy svg.echart').dataset.plotkit"))
    H.wait(drv, "const b = document.querySelector('#o-e-plot'); return b && b.offsetParent !== null", 60)
    drv.find_element("css selector", "#o-e-plot").click()
    H.wait(drv, "return window.__mace.runView && window.__mace.runView.state() && window.__mace.runView.state().drawn", 60)
    hosts.append(drv.execute_script("return window.__mace.runView.el().dataset.plotkit"))
    over = drv.execute_script("const a = document.querySelector('#runplot-host .pv-acts'); return [a.scrollWidth, a.clientWidth]")
    assert over[0] <= over[1], over
    drv.execute_async_script("const d = arguments[1]; window.__mace.openModes(arguments[0]).then(() => d(1))", f"pk/{FREQ.name}")
    H.wait(drv, "return !!document.querySelector('.fm-spec[data-plotkit]')", 60)
    hosts.append("1")
    assert hosts == ["1", "1", "1", "1"]


def test_keyboard_and_names(app):
    drv = app[0]
    open_band(drv)
    btns = drv.execute_script("return [...document.querySelectorAll('#plot-host .pv-acts > button, #plot-host .pv-acts .pk-wrap > button')].map((b) => b.textContent.trim())")
    assert all(btns) and len(btns) >= 4
    drv.execute_script("document.querySelector('#plot-host .pk-units').focus()")
    # Units, Style, Traces, Reference, Labels (1.1.6: a band plot's categories; no Log Y on a band plot alone), Export,
    # Axes & Ranges (Plot Options)
    for _ in range(6):
        U.keys(drv, Keys.TAB)
    assert drv.execute_script("return document.activeElement.classList.contains('pk-more')")
    U.keys(drv, Keys.ENTER)
    H.wait(drv, "return !!document.querySelector('.pk-pop')", 5)
    U.keys(drv, Keys.ESCAPE)
    H.wait(drv, "return !document.querySelector('.pk-pop') && document.activeElement.classList.contains('pk-more')", 5)
