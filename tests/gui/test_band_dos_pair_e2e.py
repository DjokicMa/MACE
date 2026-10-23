"""A band plot and a DOS plot of one system from two folders (1.1.6): Add DOS… on a
band plot and Add Band… on a DOS plot (next to Log Y) pick the other half from
the served folder and draw the band + DOS figure, as when both sit side by side.
The server checks the pair (the same formula, else refused; the cell and the
Fermi level, else warned), the pairing is kept per document, and Copy CLI
Command and Publication Figure carry it (mace plotting --band-dos --partner).
Firefox, the bundled example suite's layout: BAND/ and DOSS/ folders."""
import lzma
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

SUITE = H.REPO_ROOT / "examples" / "suite"
POLY = SUITE / "4-polymer-1d" / "polyacetylene"
GRAPHENE = SUITE / "3-slab-2d" / "graphene"
BAND = "poly/BAND/polyacetylene_band.BAND.DAT"
DOS = "poly/DOSS/polyacetylene_doss.DOSS.DAT"
OTHER = "graphene/DOSS/graphene_doss.DOSS.DAT"


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
    tmp = tmp_path_factory.mktemp("bdpair")
    root = H.examples_copy(tmp)
    put(POLY / "BAND", ("polyacetylene_band.BAND.DAT", "polyacetylene_band.d3", "polyacetylene_band.out"), root / "poly" / "BAND")
    put(POLY / "DOSS", ("polyacetylene_doss.DOSS.DAT", "polyacetylene_doss.d3", "polyacetylene_doss.out"), root / "poly" / "DOSS")
    put(GRAPHENE / "DOSS", ("graphene_doss.DOSS.DAT", "graphene_doss.d3", "graphene_doss.out"), root / "graphene" / "DOSS")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = None
    try:
        drv = H.browser("firefox", width=1400, height=900)
        drv.set_script_timeout(240)
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'on');"
                           "Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.plot.')).forEach((k) => localStorage.removeItem(k))")
        yield drv
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def open_plot(drv, path):
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())", path)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn === arguments[0]", 60, path)
    time.sleep(0.4)


def buttons(drv):
    return drv.execute_script("return [...document.querySelectorAll('#plot-host .pv-acts > button')].map((b) => b.textContent.trim())")


def own(drv):
    drv.find_element("css selector", "#plot-host .pk-own").click()


def two_axes(drv):
    return drv.execute_script("const l = window.__mace.plotView.el().layout; return !!(l && l.xaxis2)")


def pick(drv, path):
    H.wait(drv, "return !!document.querySelector(`#plot-pair [data-pair=\"${arguments[0]}\"]`)", 10, path)
    drv.execute_script("document.querySelector(`#plot-pair [data-pair=\"${arguments[0]}\"]`).click()", path)


def test_add_dos_from_another_folder(app):
    drv = app
    open_plot(drv, BAND)
    assert "Add DOS…" in buttons(drv) and not two_axes(drv)
    own(drv)
    H.wait(drv, "const d = document.getElementById('plot-pair'); return !!(d && d.open)", 10)
    listed = drv.execute_script("return [...document.querySelectorAll('#plot-pair [data-pair]')].map((b) => b.dataset.pair)")
    assert listed[0] == DOS and OTHER in listed and BAND not in listed, listed       # the same material first
    # another system: refused in the dialog, which stays open
    pick(drv, OTHER)
    H.wait(drv, "return /not the same system/.test(document.querySelector('#plot-pair .pv-pair-msg').textContent)", 30)
    assert drv.execute_script("return document.getElementById('plot-pair').open") is True
    pick(drv, DOS)
    H.wait(drv, "return !document.getElementById('plot-pair').open", 30)
    H.wait(drv, "const l = window.__mace.plotView.el().layout; return window.__mace.plotView.state().drawn && !!(l && l.xaxis2)", 60)
    b = buttons(drv)
    assert "Log DOS" in b and "Remove DOS" in b and "Add DOS…" not in b, b
    assert drv.execute_script("return window.__mace.plotView.state().opts.pair") == DOS
    assert "with ../DOSS/polyacetylene_doss.DOSS.DAT" in drv.execute_script("return document.querySelector('#plot-host .pv-chips').textContent")
    cli = drv.execute_script("return window.__mace.plotView.cliText()")
    assert cli.startswith("mace plotting --band-dos polyacetylene_band.BAND.DAT --partner ../DOSS/polyacetylene_doss.DOSS.DAT"), cli
    # kept per document: reopened, it is drawn with its DOS again
    open_plot(drv, BAND)
    H.wait(drv, "const l = window.__mace.plotView.el().layout; return !!(l && l.xaxis2)", 60)
    # Publication Figure: mace plotting --band-dos with the other half (--partner)
    r = drv.execute_async_script("const done = arguments[0]; window.__mace.plotView.publication().then((x) => done({ok: true, r: x}), "
                                 "(e) => done({ok: false, e: String(e.message || e)}))")
    assert r["ok"], r
    assert "--band-dos" in r["r"]["command"] and "--partner" in r["r"]["command"], r["r"]["command"]
    assert any(n.endswith(".svg") for n in r["r"]["images"]), r["r"]
    # Remove DOS: the bands alone again
    drv.execute_script("[...document.querySelectorAll('#plot-host .pk-own')].find((b) => b.textContent === 'Remove DOS').click()")
    H.wait(drv, "const l = window.__mace.plotView.el().layout; return window.__mace.plotView.state().drawn && l && !l.xaxis2", 60)
    assert "Add DOS…" in buttons(drv)
    assert not H.errors(drv), H.errors(drv)


def test_add_band_to_a_dos_plot(app):
    drv = app
    open_plot(drv, DOS)
    assert "Add Band…" in buttons(drv) and "Log Y" in buttons(drv)
    own(drv)
    H.wait(drv, "const d = document.getElementById('plot-pair'); return !!(d && d.open)", 10)
    pick(drv, BAND)
    H.wait(drv, "const l = window.__mace.plotView.el().layout; return window.__mace.plotView.state().drawn && !!(l && l.xaxis2)", 60)
    assert "Remove Band" in buttons(drv) and "Log DOS" in buttons(drv)
    cli = drv.execute_script("return window.__mace.plotView.cliText()")
    assert cli.startswith("mace plotting --band-dos polyacetylene_doss.DOSS.DAT --partner ../BAND/polyacetylene_band.BAND.DAT"), cli
    assert not H.errors(drv), H.errors(drv)
