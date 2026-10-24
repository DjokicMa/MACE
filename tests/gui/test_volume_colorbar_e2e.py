"""Color by Potential's color bar and the Volume view's picture (Firefox, the bundled example suite).

h-BN's density surface colored by its potential (the suite's CHARGE+POTENTIAL cubes) is all positive:
the bar's range is the surface's own (not from a false 0), it says "all positive on this surface" with
no sign words and no 0 tick, its ticks sit at their values, and it names the unit. Export Image works
on a Volume document and draws the bar into the PNG (at the view's right edge). The orbital picker
keeps clear of the Explorer's edge tab when the Explorer is hidden."""
import base64
import io
import lzma
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly", reason="the cube engine imports plotly")
pytest.importorskip("scipy")
PIL = pytest.importorskip("PIL.Image")

SUITE = H.REPO_ROOT / "examples" / "suite"
HBN = SUITE / "3-slab-2d" / "hBN" / "CHARGE+POTENTIAL"
POLY = SUITE / "4-polymer-1d" / "polyacetylene" / "ORBITALS"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("volbar")
    root = H.examples_copy(tmp)
    v = root / "vbar"
    v.mkdir()
    for kind in ("DENS", "POT"):
        src = HBN / f"hBN_charge+potential_{kind}.CUBE.xz"
        if not src.is_file():
            pytest.skip("the bundled example suite is not here")
        (v / f"hBN_{kind}.CUBE").write_bytes(lzma.decompress(src.read_bytes()))
    for src in POLY.iterdir() if POLY.is_dir() else ():           # the orbitals with their run (.out, .d3)
        name = src.name[:-3] if src.suffix == ".xz" else src.name
        (v / name).write_bytes(lzma.decompress(src.read_bytes()) if src.suffix == ".xz" else src.read_bytes())
    a = U.start_app(tmp, width=1400, height=900, root=root, MACE_GUI_MAX_WORKERS="2")
    try:
        yield a
    finally:
        U.stop_app(a)


def open_volume(drv, path, timeout=120):
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())", path)
    H.wait(drv, "const s = window.__mace.volumeView && window.__mace.volumeView.state(); return s && s.surfaces > 0", timeout)
    time.sleep(0.5)


def test_all_positive_potential_bar_and_its_png(app):
    drv = app[0]
    U.reset(drv, 1400, 900)
    drv.execute_script("window.__mace.settings.set('reduceMotion', 'on'); window.__mace.settings.set('volume.quality', 'low')")
    open_volume(drv, "vbar/hBN_DENS.CUBE")
    drv.execute_async_script("const done = arguments[1]; window.__mace.colorByPotential(arguments[0]).then(() => done(), (e) => done(String(e)))",
                             "vbar/hBN_POT.CUBE")
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s && s.color && s.vmin !== null", 120)
    st = drv.execute_script("return window.__mace.volumeView.state()")
    assert 0 < st["vmin"] < st["vmax"], st                        # the surface's own range, not from 0
    m = drv.execute_script("return window.__mace.volumeView.colorBar()")
    assert m["note"] == "all positive on this surface" and not m["top"] and not m["bottom"], m
    assert m["unit"] == "(Ha e⁻¹)" and all(t["v"] > 0 for t in m["ticks"]) and len(m["ticks"]) >= 3, m
    for t in m["ticks"]:
        assert t["f"] == pytest.approx((t["v"] - m["lo"]) / (m["hi"] - m["lo"]), abs=1e-9)
    # the bar as drawn: its words, no sign words, ticks in order up the bar, clear of the view's right edge
    bar = drv.execute_script("""const b = document.querySelector('#volume-host .vv-bar'), r = b.getBoundingClientRect(),
        v = document.querySelector('#viewer-volume').getBoundingClientRect();
      return {text: b.innerText, label: b.getAttribute('aria-label'), right: v.right - r.right, w: r.width,
        ticks: [...b.querySelectorAll('.vv-ticks span')].map((s) => [s.textContent, s.getBoundingClientRect().top])}""")
    assert "all positive on this surface" in bar["text"] and "negative" not in bar["text"] and "Potential" in bar["text"], bar
    assert "red = lowest" in bar["label"] and bar["right"] >= 20 and bar["w"] < 220, bar
    tops = [t[1] for t in bar["ticks"]]
    assert tops == sorted(tops, reverse=True) and "0" not in [t[0] for t in bar["ticks"]], bar
    # the gradient is the surface's own: its low end red (chemistry colors), its high end blue
    lo, hi = m["stops"][0]["c"], m["stops"][-1]["c"]
    assert int(lo[1:3], 16) > int(lo[5:7], 16) and int(hi[5:7], 16) > int(hi[1:3], 16), (lo, hi)
    U.shot(drv, "volume_colorbar_all_positive")
    # Export Image on a Volume document: the PNG holds the bar at its right edge
    drv.execute_script("window.__lastExport = null; window.__maceExportHook = (u) => { window.__lastExport = u; }")
    drv.execute_script("window.__mace.run('file.exportImage')")
    d = H.wait(drv, "const d = document.getElementById('export-image'); return d && d.open && d", 10)
    assert drv.execute_script("return document.querySelector('#export-image [data-k=legend]').closest('label').hidden") is True
    d.find_element("css selector", "[data-v=export]").click()
    url = H.wait(drv, "return window.__lastExport", 60)
    img = PIL.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    w, h = img.size
    px = img.load()
    strip = [px[x, y] for x in range(int(w * 0.7), w) for y in range(int(h * 0.3), int(h * 0.7))]
    reds = sum(1 for r, g, b in strip if r > 200 and g < 90 and b < 90)
    blues = sum(1 for r, g, b in strip if b > 200 and r < 90 and g < 90)
    assert reds > 20 and blues > 20, (reds, blues)               # the bar's two ends, drawn into the picture
    assert drv.execute_script("return document.getElementById('viewer-volume').offsetWidth") < 2000
    U.no_errors(drv)


def test_orbital_picker_clears_the_explorer_edge_tab(app):
    drv = app[0]
    U.reset(drv, 1400, 900)
    name = drv.execute_script("return (window.MACE.state.files || []).map((f) => f.path).find((p) => p.startsWith('vbar/') && p.endsWith('.molden'))")
    if not name:
        pytest.skip("no polyacetylene orbitals in the suite")
    open_volume(drv, name, 180)
    drv.execute_script("window.__mace.run('view.explorer')")
    try:
        H.wait(drv, "const e = document.querySelector('.edge-tab[data-panel=explorer]'); return e && !e.hidden && e.offsetWidth > 0", 10)
        time.sleep(0.5)
        r = drv.execute_script("""const tab = document.querySelector('.edge-tab[data-panel=explorer]').getBoundingClientRect();
          const orb = document.querySelector('#volume-host .vv-orb');
          const kids = [...orb.querySelectorAll('label, span, button, input, select, h3, p')].filter((e) => e.offsetWidth > 0)
            .map((e) => e.getBoundingClientRect()).filter((b) => b.bottom > tab.top && b.top < tab.bottom);
          return {tab: tab.right, left: Math.min(...kids.map((b) => b.left)), n: kids.length}""")
        assert r["n"] > 0 and r["left"] >= r["tab"], r
    finally:
        drv.execute_script("window.__mace.run('view.explorer')")
    U.no_errors(drv)
