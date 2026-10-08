"""Export with a resolution and a physical size, and SVG shading (plan E9), in
headless Firefox with real clicks on a fresh profile; the shipped demo
folder only (runs in CI).

A PNG of 3.5 in at 300 DPI is 1050 px wide and says 300 DPI (read back with
PIL from the real download); a band plot at 600 DPI and 3.5 in is 2100 px
with its pHYs; SVG Flat has no gradients, Shaded one per element with every
atom filled from one; a fresh profile opens the dialog at 150 DPI."""
import io
import time
import xml.etree.ElementTree as ET

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
PIL = pytest.importorskip("PIL.Image")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.support.ui import Select  # noqa: E402

SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export_dpi")
    root = H.examples_copy(tmp)
    # a band plot: a real corpus BAND.DAT under a neutral name (that test skips without the corpus)
    import shutil
    corpus = H.REPO_ROOT / "test" / "BAND"
    bands = sorted(corpus.glob("*.BAND.DAT"), key=lambda p: p.stat().st_size) if corpus.is_dir() else []
    if bands:
        (root / "bands").mkdir()
        shutil.copyfile(bands[0], root / "bands" / "run.BAND.DAT")
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    for f in app[5].glob("*"):
        f.unlink()
    return app


def click(drv, el):
    ActionChains(drv).move_to_element(el).click().perform()


def downloaded(dl, name, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        p = dl / name
        if p.exists() and p.stat().st_size > 0 and not (dl / (name + ".part")).exists():
            time.sleep(0.2)
            return p.read_bytes()
        time.sleep(0.2)
    raise AssertionError(f"{name} was not downloaded: {[f.name for f in dl.iterdir()]}")


def open_dialog(drv):
    drv.execute_script("window.__mace.run('file.exportImage')")
    return H.wait(drv, "const d = document.getElementById('export-image'); return d && d.open && d", 10)


def test_fresh_profile_opens_at_150_dpi_and_png_says_its_dpi(fresh):
    drv, dl = fresh[0], fresh[5]
    U.open_doc(drv, "structures/MgO.cif")
    d = open_dialog(drv)
    assert Select(d.find_element(By.CSS_SELECTOR, '[data-k="dpisel"]')).first_selected_option.text == "150 DPI"
    Select(d.find_element(By.CSS_SELECTOR, '[data-k="size"]')).select_by_value("physical")
    pw = H.wait(drv, "const e = document.querySelector('#export-image [data-k=pw]'); return e && e.offsetWidth > 0 && e", 10)
    time.sleep(0.2)
    pw.clear()
    pw.send_keys("3.5")
    Select(d.find_element(By.CSS_SELECTOR, '[data-k="dpisel"]')).select_by_value("300")
    note = d.find_element(By.CSS_SELECTOR, '[data-k="px"]').text
    assert note.startswith("1050 × ") and "3.50 ×" in note and "300 DPI" in note, note
    name = d.find_element(By.CSS_SELECTOR, '[data-k="name"]').get_attribute("value")
    click(drv, d.find_element(By.CSS_SELECTOR, '[data-v="export"]'))
    img = PIL.open(io.BytesIO(downloaded(dl, name)))
    assert img.size[0] == 1050 and tuple(round(x) for x in img.info["dpi"]) == (300, 300), (img.size, img.info)
    U.no_errors(drv)


def test_svg_flat_and_shaded(fresh):
    drv, dl = fresh[0], fresh[5]
    U.open_doc(drv, "structures/MgO.cif")
    texts = {}
    for shading in ("flat", "shaded"):
        for f in dl.glob("*"):
            f.unlink()
        d = open_dialog(drv)
        Select(d.find_element(By.CSS_SELECTOR, '[data-k="format"]')).select_by_value("svg")
        sel = d.find_element(By.CSS_SELECTOR, '[data-k="shading"]')
        assert sel.is_displayed()
        Select(sel).select_by_value(shading)
        name = d.find_element(By.CSS_SELECTOR, '[data-k="name"]').get_attribute("value")
        click(drv, d.find_element(By.CSS_SELECTOR, '[data-v="export"]'))
        texts[shading] = downloaded(dl, name).decode()
    flat, shaded = ET.fromstring(texts["flat"]), ET.fromstring(texts["shaded"])
    assert not flat.findall(f".//{SVG}radialGradient")
    grads = shaded.findall(f".//{SVG}radialGradient")
    circles = shaded.findall(f".//{SVG}circle")
    fills = {c.get("fill") for c in circles}
    assert len(grads) == len(fills) == 2                     # Mg and O
    assert all(f.startswith("url(#") for f in fills) and {f[5:-1] for f in fills} == {g.get("id") for g in grads}
    U.no_errors(drv)


def test_band_plot_at_600_dpi(fresh):
    pytest.importorskip("plotly")
    drv, dl, root = fresh[0], fresh[5], fresh[2]
    bands = sorted((root / "bands").glob("*.BAND.DAT")) if (root / "bands").is_dir() else []
    if not bands:
        pytest.skip("test/ corpus not present (gitignored)")
    U.open_doc(drv, f"bands/{bands[0].name}", wait_atoms=False)
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.drawn", 60)
    # the plot toolbar's Export menu (plotkit) › Image…
    click(drv, drv.find_element(By.CSS_SELECTOR, '#plot-host .pk-export'))
    click(drv, H.wait(drv, "return document.querySelector('#plot-host .pk-menu [data-v=image]')", 10))
    d = H.wait(drv, "const d = document.getElementById('export-image'); return d && d.open && d", 10)
    Select(d.find_element(By.CSS_SELECTOR, '[data-k="size"]')).select_by_value("physical")
    pw = H.wait(drv, "const e = document.querySelector('#export-image [data-k=pw]'); return e && e.offsetWidth > 0 && e", 10)
    time.sleep(0.2)
    pw.clear()
    pw.send_keys("3.5")
    Select(d.find_element(By.CSS_SELECTOR, '[data-k="dpisel"]')).select_by_value("600")
    name = d.find_element(By.CSS_SELECTOR, '[data-k="name"]').get_attribute("value")
    click(drv, d.find_element(By.CSS_SELECTOR, '[data-v="export"]'))
    img = PIL.open(io.BytesIO(downloaded(dl, name, 40)))
    assert img.size[0] == 2100 and tuple(round(x) for x in img.info["dpi"]) == (600, 600), (img.size, img.info)
    U.no_errors(drv)
