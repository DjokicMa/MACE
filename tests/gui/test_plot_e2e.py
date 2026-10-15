"""Band and DOS plots as documents (plan T8; spec §7 Phase 5 criterion 2, §8.5).

Synthetic BAND.DAT / DOSS.DAT files (the writers of test_gui_results and
test_ipdos_readers: CRYSTAL's layout, made-up numbers) in a demo copy.
plotly comes from the plotly package, only when the first plot opens, once;
nothing loads from another origin. The band document has its k-path ticks
(Γ for G), a hover readout, the Fermi line that drags the reference, Plot
options with their own undo and Reset, kept across a reload."""
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
sys.path.insert(0, str(H.REPO_ROOT / "tests"))
from test_gui_results import write_band  # noqa: E402
from test_ipdos_readers import write_doss  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p5plot")
    root = H.examples_copy(tmp)
    (root / "results").mkdir()
    write_band(root / "results")
    write_doss(root / "results", material="mgo")
    # a crystal's IRSPEC.DAT (CRYSTAL23 manual p.453): cm-1, nm, 4 absorbances, 3 reflectances
    (root / "results" / "agcl_IRSPEC.DAT").write_text("".join(
        "  ".join(f"{v:.5E}" for v in [10.0 * k, 1e6 / k] + [c + k / 100 for c in range(3, 10)]) + "\n" for k in range(1, 60)))
    (root / "spin").mkdir()
    write_band(root / "spin", stem="fe_band", spin=True)
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    app[0].execute_script("Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.plot.')).forEach((k) => localStorage.removeItem(k))")
    return app


def resources(drv):
    return drv.execute_script("return performance.getEntriesByType('resource').map((r) => ({name: r.name, size: r.transferSize, enc: r.encodedBodySize}))")


def open_plot(drv, path):
    U.open_doc(drv, path, wait_atoms=False)
    try:
        H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn === arguments[0]", 60, path)
    except AssertionError:
        raise AssertionError(drv.execute_script("const v = window.__mace.plotView; return [v && v.state(), v && v.el().dataset.drawn,"
                                                " document.querySelector('#plot-host .pv-note').textContent, window.__mace.activeDoc() && window.__mace.activeDoc().title, window.__maceErrors]"))
    time.sleep(0.3)


def test_plotly_loads_lazily_once_and_locally(fresh):
    drv, base = fresh[0], fresh[1]
    H.load(drv, base)
    U.open_doc(drv, "structures/MgO.cif")
    assert not [r for r in resources(drv) if "plotly" in r["name"]]
    open_plot(drv, "results/mgo_band.BAND.DAT")
    pl = [r for r in resources(drv) if "plotly" in r["name"]]
    assert len(pl) == 1 and pl[0]["name"].startswith(base) and "/static/v-" in pl[0]["name"]
    # the bench's plotly_lazy budget: 1.4 MB gzipped with 5% slack. plotly 6.0 sends 1.41 MB,
    # plotly 7.1 1.47 MB (its plotly.min.js grew), so the slack matters on new installs
    budget = 1.4 * 2**20 * 1.05
    assert 0 < pl[0]["size"] <= budget or pl[0]["enc"] <= budget, pl
    open_plot(drv, "results/mgo_doss.DOSS.DAT")
    assert len([r for r in resources(drv) if "plotly" in r["name"]]) == 1
    assert all(r["name"].startswith(base) or r["name"].startswith("data:") or r["name"].startswith("blob:") for r in resources(drv))
    U.no_errors(drv)


def test_band_document_ticks_fermi_and_readout(fresh):
    drv = fresh[0]
    open_plot(drv, "results/mgo_band.BAND.DAT")
    lay = drv.execute_script("const p = document.querySelector('#plot-host .pv-plot'); return {ticks: p.layout.xaxis.ticktext, "
                             "shapes: p.layout.shapes.map((s) => s.y0), names: p.data.map((t) => t.name), y2: !!p.layout.xaxis2}")
    assert lay["ticks"][0] == "Γ" and "Γ" in lay["ticks"]
    fermi = drv.execute_script("return window.__mace.plotView.figure().layout.shapes[0].y0")
    assert lay["y2"]                                     # the DOS partner on the same energy axis
    assert len([n for n in lay["names"] if n in ("bands", "spin up (α)")]) == 1 and "spin down (β)" not in lay["names"]
    # a real pointer over a band point (Plotly.Fx.hover sets the hover data without the event)
    xy = drv.execute_script("""const p = document.querySelector('#plot-host .pv-plot'), L = p._fullLayout, t = p.data[0];
        const r = p.getBoundingClientRect();
        return [r.left + L.xaxis._offset + L.xaxis.l2p(t.x[2]), r.top + L.yaxis._offset + L.yaxis.l2p(t.y[2])]""")
    from selenium.webdriver.common.action_chains import ActionChains
    body = drv.find_element("tag name", "body")
    ActionChains(drv).move_to_element_with_offset(body, 0, 0).perform()
    drv.execute_script("window.scrollTo(0, 0)")
    ActionChains(drv).move_by_offset(int(xy[0]) - drv.execute_script("return document.body.getBoundingClientRect().width / 2"),
                                     int(xy[1]) - drv.execute_script("return document.body.getBoundingClientRect().height / 2")).perform()
    H.wait(drv, "return document.querySelector('#plot-host .pv-readout').textContent.includes('E − E_F')", 5)
    rd = drv.find_element("css selector", "#plot-host .pv-readout").text
    assert "band 1" in rd and "k:" in rd and "eV" in rd, rd
    U.shot(drv, "plot_band_dos_light")
    drv.execute_script("window.__mace.settings.set('theme', 'dark')")
    time.sleep(0.6)
    U.shot(drv, "plot_band_dos_dark")
    drv.execute_script("window.__mace.settings.set('theme', 'light')")


def test_spin_polarized_bands_draw_both_spins(fresh):
    drv = fresh[0]
    open_plot(drv, "spin/fe_band.BAND.DAT")
    names = drv.execute_script("return document.querySelector('#plot-host .pv-plot').data.map((t) => t.name)")
    assert "spin up (α)" in names and "spin down (β)" in names


def test_zoom_shares_the_energy_axis_and_fits_the_dos(fresh):
    drv = fresh[0]
    open_plot(drv, "results/mgo_band.BAND.DAT")
    r0 = drv.execute_script("return document.querySelector('#plot-host .pv-plot').layout.xaxis2.range")
    drv.execute_script("const p = document.querySelector('#plot-host .pv-plot'); return Plotly.relayout(p, {'yaxis.range': [-2, 1]})")
    time.sleep(0.5)
    st = drv.execute_script("const p = document.querySelector('#plot-host .pv-plot'); return {y: p.layout.yaxis.range, x2: p.layout.xaxis2.range}")
    assert st["y"] == [-2, 1]                  # one y axis for the bands and the DOS
    assert st["x2"] != r0                     # the DOS axis refits to the states in the new window


def test_fermi_drag_undo_reset_and_reload(fresh):
    drv, base = fresh[0], fresh[1]
    open_plot(drv, "results/mgo_band.BAND.DAT")
    fermi = drv.execute_script("return window.__mace.plotView.figure().layout.shapes[0].y0")
    drv.execute_script("const p = document.querySelector('#plot-host .pv-plot'); return Plotly.relayout(p, {'shapes[0].y0': arguments[0], 'shapes[0].y1': arguments[0]})",
                       fermi + 0.25)
    H.wait(drv, "return Math.abs(window.__mace.plotView.state().opts.shift - 0.25) < 1e-6", 10)
    assert "E" in drv.find_element("css selector", "#plot-host .pv-chips").text and "0.25" in drv.find_element("css selector", "#plot-host .pv-chips").text
    assert drv.execute_script("return document.querySelector('#po-shift').value") == "0.25"
    U.shot(drv, "plot_fermi_shift")
    drv.execute_script("document.querySelector('#plot-host .pv-plot').focus()")
    U.keys(drv, Keys.CONTROL, "z")
    H.wait(drv, "return window.__mace.plotView.state().opts.shift === 0", 10)
    U.keys(drv, Keys.CONTROL, Keys.SHIFT, "z")
    H.wait(drv, "return window.__mace.plotView.state().opts.shift === 0.25", 10)
    # the options survive a reload
    H.load(drv, base)
    open_plot(drv, "results/mgo_band.BAND.DAT")
    assert drv.execute_script("return window.__mace.plotView.state().opts.shift") == 0.25
    drv.execute_script("window.__mace.run('analysis.resetPlot')")
    H.wait(drv, "return window.__mace.plotView.state().opts.shift === 0", 10)
    o = drv.execute_script("return window.__mace.plotView.state().opts")
    assert (o["emin"], o["emax"], o["spin"]) == (-6, 6, "both")
    U.no_errors(drv)


def test_export_png_svg_and_data(fresh):
    drv, base, root, xdg, proc, downloads = fresh
    open_plot(drv, "results/mgo_band.BAND.DAT")
    for f in downloads.iterdir():
        f.unlink()
    for act in ("png", "svg", "csv"):
        # the export buttons moved into the plotkit toolbar's Export menu (PLAN_analysis C3)
        drv.find_element("css selector", "#plot-host .pk-export").click()
        drv.find_element("css selector", f".pk-menu [data-v={act}]").click()
    end = time.time() + 20
    while time.time() < end and len([f for f in downloads.iterdir() if not f.name.endswith(".part")]) < 3:
        time.sleep(0.2)
    names = sorted(f.name for f in downloads.iterdir())
    assert any(n.endswith(".png") for n in names) and any(n.endswith(".svg") for n in names) and any(n.endswith("_data.csv") for n in names), names
    svg = next(f for f in downloads.iterdir() if f.name.endswith(".svg")).read_text()
    assert svg.lstrip().startswith("<svg") and "http" not in svg.split(">")[0].replace("http://www.w3.org", "")


def test_plot_refits_a_pane_resized_while_hidden(fresh):
    """plotly's own "responsive" follows window resizes only: a window resized while a structure
    tab is showing used to leave the plot at its old size when its tab came back."""
    drv = fresh[0]
    fitted = ("const el = window.__mace.plotView.el(), fl = el._fullLayout;"
              " return [el.clientWidth, el.clientHeight, fl && fl.width, fl && fl.height]")
    try:
        open_plot(drv, "results/mgo_band.BAND.DAT")
        U.open_doc(drv, "structures/MgO.cif")
        H.set_viewport(drv, 1920, 1080)
        time.sleep(0.6)
        U.open_doc(drv, "results/mgo_band.BAND.DAT", wait_atoms=False)
        H.wait(drv, "const el = window.__mace.plotView.el(), fl = el._fullLayout;"
                    " return !!fl && el.clientWidth > 1000 && Math.abs(el.clientWidth - fl.width) <= 4", 10)
        w, h, fw, fh = drv.execute_script(fitted)
        assert w > 1000 and abs(w - fw) <= 4 and abs(h - fh) <= 4, (w, h, fw, fh)
        U.no_errors(drv)
    finally:
        H.set_viewport(drv, 1280, 800)


def test_crystal_irspec_columns_and_export(fresh):
    """IRSPEC.DAT's columns are named as the manual names them (no "transmittance"):
    the absorbances and reflectances are components, the wavelength goes to the export."""
    drv, base, root, xdg, proc, downloads = fresh
    open_plot(drv, "results/agcl_IRSPEC.DAT")
    opts = drv.execute_script("return [...document.querySelectorAll('#po-comp option')].map((o) => o.value)")
    assert opts == ["all", "absorbance", "absorbance_classical", "absorbance_rayleigh_1", "absorbance_rayleigh_2",
                    "reflectance_1", "reflectance_2", "reflectance_3"], opts
    assert drv.execute_script("return document.querySelector('#po-comp').value") == "absorbance"
    fig = drv.execute_script("return window.__mace.plotView.figure()")
    assert len(fig["traces"]) == 1 and fig["layout"]["yaxis"]["title"]["text"] == "Absorbance"
    drv.execute_script("const c = document.querySelector('#po-comp'); c.value = 'reflectance_2'; c.dispatchEvent(new Event('change', {bubbles: true}))")
    H.wait(drv, "return window.__mace.plotView.figure().layout.yaxis.title.text === 'Reflectance'", 20)
    for f in downloads.iterdir():
        f.unlink()
    drv.find_element("css selector", "#plot-host .pk-export").click()
    drv.find_element("css selector", ".pk-menu [data-v=csv]").click()
    end = time.time() + 20
    while time.time() < end and not [f for f in downloads.iterdir() if f.name.endswith("_data.csv")]:
        time.sleep(0.2)
    csv = next(f for f in downloads.iterdir() if f.name.endswith("_data.csv")).read_text().splitlines()
    assert csv[0].split(",") == ["wavenumber_cm-1", "wavelength_nm", "absorbance", "absorbance_classical", "absorbance_rayleigh_1",
                                 "absorbance_rayleigh_2", "reflectance_1", "reflectance_2", "reflectance_3"], csv[0]
    assert [float(v) for v in csv[1].split(",")[:4]] == [10.0, 1e6, 3.01, 4.01]
    U.no_errors(drv)
