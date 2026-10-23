"""Log Y is a DOS axis (1.1.6). Band energies are signed, so a band plot alone
has no Log Y, a band + DOS plot's button reads Log DOS and makes the DOS axis
logarithmic while the energy axis stays linear, and a DOS plot alone keeps its
Log Y. Publication Figure and Copy CLI Command carry it as --log-dos.
Firefox, polyacetylene of the bundled example suite."""
import lzma
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

RUN = H.REPO_ROOT / "examples" / "suite" / "4-polymer-1d" / "polyacetylene"
BAND = ("polyacetylene_band.BAND.DAT", "polyacetylene_band.d3", "polyacetylene_band.out")
DOSS = ("polyacetylene_doss.DOSS.DAT", "polyacetylene_doss.d3", "polyacetylene_doss.out")


def put(src_dir, names, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        xz, plain = src_dir / (name + ".xz"), src_dir / name
        if xz.is_file():
            (dst_dir / name).write_bytes(lzma.decompress(xz.read_bytes()))
        elif plain.is_file():
            (dst_dir / name).write_bytes(plain.read_bytes())
        else:
            pytest.skip("the bundled example suite is not here")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("logdos")
    root = H.examples_copy(tmp)
    put(RUN / "BAND", BAND, root / "band")
    put(RUN / "DOSS", DOSS, root / "dos")
    put(RUN / "BAND", BAND, root / "both")
    put(RUN / "DOSS", DOSS, root / "both")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = None
    try:
        drv = H.browser("firefox", width=1400, height=900)
        drv.set_script_timeout(120)
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'on')")
        yield drv
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def open_plot(drv, path):
    drv.execute_script("window.__mace.closeAll(); Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.plot.'))"
                       ".forEach((k) => localStorage.removeItem(k))")
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())", path)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn === arguments[0]", 60, path)
    time.sleep(0.4)


LOGY = "const b = document.querySelector('#plot-host .pk-logy'); return b && {text: b.textContent, title: b.title, pressed: b.getAttribute('aria-pressed')}"
AXES = """const l = window.__mace.plotView.el().layout;
return {y: l.yaxis && l.yaxis.type, x: l.xaxis && l.xaxis.type, x2: l.xaxis2 ? l.xaxis2.type : null}"""
# the DOS traces' fills: a log DOS total is drawn unfilled (review 1.1.6 E3: plotly joined the fill across the gap)
FILLS = "return window.__mace.plotView.el().data.filter((t) => t.meta === 'dos').map((t) => t.fill || 'none')"


def test_a_band_plot_alone_has_no_log_y(app):
    drv = app
    open_plot(drv, "band/polyacetylene_band.BAND.DAT")
    assert drv.execute_script(LOGY) is None
    # a ylog saved by an older page draws the energies linear
    drv.execute_script("window.__mace.plotView.setOpt({ylog: true}, 'Log Y')")
    time.sleep(0.6)
    assert drv.execute_script(AXES)["y"] == "linear"
    assert not H.errors(drv), H.errors(drv)


def test_band_and_dos_log_dos_is_the_dos_axis(app):
    drv = app
    open_plot(drv, "both/polyacetylene_band.BAND.DAT")
    b = drv.execute_script(LOGY)
    assert b["text"] == "Log DOS" and "band energies stay linear" in b["title"] and b["pressed"] == "false", b
    assert "tozerox" in drv.execute_script(FILLS)
    drv.find_element("css selector", "#plot-host .pk-logy").click()
    H.wait(drv, "const b = document.querySelector('#plot-host .pk-logy'); return b && b.getAttribute('aria-pressed') === 'true'", 10)
    time.sleep(0.6)
    ax = drv.execute_script(AXES)
    assert ax["x2"] == "log" and ax["y"] == "linear", ax
    assert set(drv.execute_script(FILLS)) == {"none"}
    cli = drv.execute_script("return window.__mace.plotView.cliText()")
    assert "--band-dos" in cli and "--log-dos" in cli, cli
    assert not H.errors(drv), H.errors(drv)


def test_a_dos_plot_keeps_its_log_y(app):
    drv = app
    open_plot(drv, "dos/polyacetylene_doss.DOSS.DAT")
    b = drv.execute_script(LOGY)
    assert b["text"] == "Log Y", b
    assert "tozerox" in drv.execute_script(FILLS)
    drv.find_element("css selector", "#plot-host .pk-logy").click()
    time.sleep(0.8)
    ax = drv.execute_script(AXES)
    assert ax["x"] == "log" and ax["y"] == "linear", ax
    assert set(drv.execute_script(FILLS)) == {"none"}
    assert "--dos" in drv.execute_script("return window.__mace.plotView.cliText()") and \
        "--log-dos" in drv.execute_script("return window.__mace.plotView.cliText()")
    assert not H.errors(drv), H.errors(drv)
