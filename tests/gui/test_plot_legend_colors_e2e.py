"""A band + DOS plot's legend keys are drawn in the colors and line styles of the traces they name (Marcus
2026-10-18): with β dashed off the Spin keys are the pair's two colors, both solid (not two black lines);
a closed-shell semiconductor with Band style Ink beside a DOS mirrored in the pair gets a Bands group of
its own next to the DOS's Spin group.
"""
import shutil
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

FE = H.REPO_ROOT / "tests" / "data" / "spin_fe"
KEYS = ["plots.style", "plots.style.band", "plots.spinColors", "plots.spinUp", "plots.spinDown", "plots.spinDownDash", "plots.bandStyle",
        "plots.bandColor", "plots.dosColors", "plots.spinDos", "plots.dosTotal", "plots.dosSmooth", "theme"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("legendcolors")
    root = H.examples_copy(tmp)
    (root / "fe").mkdir()
    for f in FE.iterdir():
        shutil.copy(f, root / "fe")
    # a closed-shell semiconductor (a gap at E_F) with its one-block DOS
    from test_dos_spin_window import write_doss_spin
    from test_gui_results import write_band_crystal
    semi = root / "semi"
    semi.mkdir()
    write_band_crystal(semi)
    head, tail = (semi / "gra_band.BAND.DAT").read_text().split("@ XAXIS TICK SPEC", 1)
    (semi / "gra_band.BAND.DAT").write_text(head.replace("NSPIN     2", "NSPIN     1") + "@ XAXIS TICK SPEC" + tail)
    doss = write_doss_spin(semi).read_text().split("&\n")[0].replace("NSPIN     2", "NSPIN     1")
    (semi / "gra_doss.DOSS.DAT").write_text(doss + "&\n")
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def drv(app, request):
    d = app[0]
    U.reset(d)
    d.execute_script("Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.plot.')).forEach((k) => localStorage.removeItem(k));"
                     "const S = window.__mace.settings, D = S.defaults();"
                     "for (const k of arguments[0]) if (k in D) S.set(k, D[k]);"
                     "S.set('theme', 'light')", KEYS)
    failed = request.session.testsfailed
    yield d
    if request.session.testsfailed > failed:
        H.save_failure(d, request.node.name)


def open_plot(drv, path):
    U.open_doc(drv, path, wait_atoms=False)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn === arguments[0]", 90, path)
    time.sleep(0.3)


def change(drv, ident, value=None, checked=None):
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = '';"
                       "const i = document.getElementById(arguments[0]); if (arguments[2] !== null) i.checked = arguments[2]; else i.value = arguments[1];"
                       "i.dispatchEvent(new Event('change', {bubbles: true}))", ident, value, checked)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn !== ''", 30)
    time.sleep(0.3)


# the drawn traces' line colors and the legend as rendered: each line key's name, group title, swatch stroke and dash
LEGEND = """const norm = (x) => { const c = document.createElement('canvas').getContext('2d'); c.fillStyle = '#123456'; c.fillStyle = x; return c.fillStyle; };
  const p = document.querySelector('#plot-host .pv-plot'), by = {};
  p.data.forEach((t) => { if (t.meta === 'up' || t.meta === 'down' || (t.meta === 'dos' && /^Total/.test(t.name)))
    by[t.meta === 'up' || t.meta === 'down' ? t.meta : t.name] = norm(t.line.color); });
  const keys = [...p.querySelectorAll('.legend .traces')].map((g) => {
    const d = g.__data__ && g.__data__[0] && g.__data__[0].trace, ln = g.querySelector('.legendlines path');
    return d && ln ? {name: d.name, meta: d.meta, group: d.legendgroup || null,
      stroke: norm(ln.style.stroke), dash: (ln.style.strokeDasharray || 'none').replace(/px/g, '')} : null; }).filter((k) => k && /^key-(spin|band)/.test(k.meta));
  const titles = [...p.querySelectorAll('.legend .legendtitletext, .legend .legendgrouptitle, .legend .groups text')].map((t) => t.textContent);
  return {by, keys, titles};"""


def test_legend_keys_take_the_drawn_colors_and_a_bands_group(drv):
    open_plot(drv, "fe/fe_band.BAND.DAT")
    change(drv, "po-bdash", checked=False)
    f = drv.execute_script(LEGEND)
    spin = [k for k in f["keys"] if k["meta"] == "key-spin"]
    assert [k["name"] for k in spin] == ["spin up (α)", "spin down (β)"], f
    # the swatches are the traces' colors (bands and DOS total alike), two different colors, no dash
    assert spin[0]["stroke"] == f["by"]["up"] == f["by"]["Total ↑"], f
    assert spin[1]["stroke"] == f["by"]["down"] == f["by"]["Total ↓"], f
    assert spin[0]["stroke"] != spin[1]["stroke"] and all(k["dash"] in ("none", "") for k in spin), f
    assert not any(k["meta"] == "key-band" for k in f["keys"]), f
    U.no_errors(drv)
    # a closed-shell semiconductor, Band style Ink, its DOS mirrored in the pair: a Bands group beside Spin
    open_plot(drv, "semi/gra_band.BAND.DAT")
    change(drv, "po-bstyle", "ink")
    change(drv, "po-dspin", "mirrored")
    f = drv.execute_script(LEGEND)
    spin = [k for k in f["keys"] if k["meta"] == "key-spin"]
    bands = [k for k in f["keys"] if k["meta"] == "key-band"]
    assert [k["stroke"] for k in spin] == [f["by"]["Total ↑"], f["by"]["Total ↓"]] and spin[0]["stroke"] != spin[1]["stroke"], f
    assert len(bands) == 1 and bands[0]["group"] == "bands" and bands[0]["stroke"] == f["by"]["up"], f
    assert bands[0]["stroke"] not in (spin[0]["stroke"], spin[1]["stroke"]), f
    assert drv.execute_script("return [...document.querySelectorAll('#plot-host .pv-plot .legend text')].some((t) => t.textContent === 'Bands')"), f
    U.no_errors(drv)
