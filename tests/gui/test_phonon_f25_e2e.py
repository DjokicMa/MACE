"""A real phonon fort.25 (tests/data/f25: CRYSTAL23 diamond, SCELPHONO 2x2x2, FREQCALC BANDS + PDOS) in the
browser: it opens as a band plot with its DOS from the same file, in frequencies, with no Fermi level."""
import shutil
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

DATA = Path(__file__).resolve().parents[1] / "data" / "f25"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("phonon_f25")
    root = H.examples_copy(tmp)
    (root / "ph").mkdir()
    for name in ("dia_phonon_f25.fort.25", "dia_phonon_f25.d12"):
        shutil.copy2(DATA / name, root / "ph" / name)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = H.firefox(1280, 800, download_dir=tmp)
    H.load(drv, base)
    try:
        yield drv
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def test_the_phonon_fort25_plots_as_frequencies(app):
    drv = app
    chip = drv.execute_script("return [...document.querySelectorAll('#file-list .file')].filter((b) => b.dataset.path === 'ph/dia_phonon_f25.fort.25').map((b) => b.querySelector('.kb').textContent)[0]")
    assert chip == "BAND"
    row = drv.find_element("css selector", '#file-list .file[data-path="ph/dia_phonon_f25.fort.25"]')
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", row)
    row.click()
    H.wait(drv, "return window.__mace.plotView && window.__mace.plotView.state() && window.__mace.plotView.state().drawn", 60)
    fig = drv.execute_script("return window.__mace.plotView.figure()")
    assert fig["layout"]["yaxis"]["title"]["text"] == "Frequency (cm<sup>−1</sup>)"
    assert "DOS" in fig["layout"]["xaxis2"]["title"]["text"]          # the DOS of the same file beside the bands
    assert not drv.execute_script("return (window.__mace.plotView.el().layout.shapes || []).some((s) => /fermi/.test(s.name || ''))")
    opts = drv.execute_script("return window.__mace.plotView.state().opts")
    assert opts["unit"] == "cm-1"
    units = drv.execute_script("document.querySelector('#plot-host .pk-units').click();"
                               "const v = [...document.querySelectorAll('.pk-menu button')].map((b) => b.dataset.v); document.querySelector('.pk-menu').close(); return v")
    assert units == ["cm-1", "THz", "meV"]
    cats = drv.execute_script("return [...document.querySelectorAll('#plot-host .pk-cat')].map((b) => b.dataset.cat)")
    assert cats == ["style", "traces", "ref", "labels"]
