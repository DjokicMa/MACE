"""Two edges of 1.1.6's Explorer and Plot Gallery (review), in Firefox and
Chromium:

* Analysis › Plot Gallery… run again on the same set goes back to its open
  gallery instead of opening another "Gallery: This folder" tab (another set
  still opens its own);
* hiding every Explorer section (View › Explorer Sections) leaves a note with
  Show All Sections, not an empty sidebar."""
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")


@pytest.fixture(scope="module", params=["firefox", "chromium"])
def drv(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(f"edges_{request.param}")
    root = H.examples_copy(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg")
    d = None
    try:
        d = H.browser(request.param, width=1280, height=900)
        H.load(d, base)
        yield d
    finally:
        if d is not None:
            d.quit()
        proc.terminate()
        proc.wait(timeout=15)


def galleries(drv):
    return drv.execute_script("return window.__mace.docs().filter((d) => d.kind === 'gallery').length")


def test_plot_gallery_reuses_the_open_gallery_of_the_same_set(drv):
    drv.execute_script("window.__mace.closeAll()")
    for _ in range(3):
        drv.execute_script("window.__mace.run('analysis.gallery')")
        H.wait(drv, "const a = window.__mace.activeDoc(); return a && a.kind === 'gallery'", 20)
        time.sleep(0.3)
    assert galleries(drv) == 1
    # away and back: still one, and it is the active one
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d(1), () => d(0))", "structures/MgO.cif")
    drv.execute_script("window.__mace.run('analysis.gallery')")
    H.wait(drv, "const a = window.__mace.activeDoc(); return a && a.kind === 'gallery'", 20)
    assert galleries(drv) == 1
    # another set (a folder) is a gallery of its own; asking it again with another kind switches that one
    drv.execute_script("window.__mace.openGallery({source: {type: 'folder', folder: 'structures'}, kind: 'structure', label: 'structures'})")
    H.wait(drv, "return window.__mace.docs().filter((d) => d.kind === 'gallery').length === 2", 20)
    drv.execute_script("window.__mace.openGallery({source: {type: 'folder', folder: 'structures'}, kind: 'scf', label: 'structures'})")
    H.wait(drv, "const v = window.__mace.galleryView; return v && v.state().kind === 'scf'", 20)
    assert galleries(drv) == 2
    assert not H.errors(drv), H.errors(drv)


def test_hiding_every_explorer_section_leaves_a_way_back(drv):
    drv.execute_script("window.__mace.closeAll(); window.__mace.sidebar().reset()")
    time.sleep(0.3)
    assert not drv.execute_script("const h = document.querySelector('#explorer .side-empty'); return !!h && !h.hidden")
    for k in ("editors", "files", "db", "recent"):
        drv.execute_script("window.__mace.run('view.sidebar.show.' + arguments[0])", k)
        time.sleep(0.15)
    H.wait(drv, "const h = document.querySelector('#explorer .side-empty'); return !!h && !h.hidden", 5)
    hint = drv.find_element("css selector", "#explorer .side-empty")
    assert "hidden" in hint.text and hint.is_displayed()
    hint.find_element("css selector", "button").click()
    H.wait(drv, "return !document.getElementById('files-section').hidden", 5)
    assert drv.execute_script("const h = document.querySelector('#explorer .side-empty'); return !h || h.hidden")
    assert not H.errors(drv), H.errors(drv)
