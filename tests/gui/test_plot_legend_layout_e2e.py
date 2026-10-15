"""The band + DOS plot's legends sit outside the data, asserted on the rendered
boxes (Firefox and Chromium).

At UI scale 110 % a spin-polarized graphite BAND + DOSS plot drew one legend
across both panels, in two rows INSIDE the plot: it covered the bands at the
top of the band panel, the y tick "6" and the DOS curves at the top right, and
listed every projection twice (C (s) ↑ ... Total ↑, C (s) ↓ ... Total ↓). The
subtitle above it was a three-line sentence.

Now the band legend (α/β, or "α = β (non-magnetic)") sits above the band panel
and the DOS legend (each projection once, plus one spin key when the DOS has
both spins) to the right of the DOS panel, in margins plotly sizes from the
legends' rendered boxes; the subtitle is one short line with the rest in its
help tooltip. Checked at UI scale 100/110/125 % and window widths
900/1366/1920 for a non-magnetic and a magnetic spin-polarized run, a
closed-shell run and a DOS with many projections.
"""
import os
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
from test_dos_spin_window import write_doss_spin  # noqa: E402
from test_gui_results import write_band, write_band_crystal  # noqa: E402
from test_plot_legend_outside import MANY, write_doss_many  # noqa: E402

@pytest.fixture(scope="module", params=["firefox", "chromium"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("legend")
    root = H.examples_copy(tmp)
    (root / "gr").mkdir()                                     # spin-polarized, alpha = beta
    write_band_crystal(root / "gr", noise=1e-4)
    write_doss_spin(root / "gr")
    (root / "mag").mkdir()                                    # spin-polarized, magnetic
    write_band_crystal(root / "mag", stem="fe_band", beta_shift=0.002)
    write_doss_spin(root / "mag", material="fe", magnetic=True)
    (root / "cl").mkdir()                                     # closed shell, ten projections
    write_band(root / "cl", stem="mgo_band")
    write_doss_many(root / "cl")
    real = H.firefox
    if request.param == "chromium":
        H.firefox = lambda w, h, download_dir=None, **kw: H.chromium(w, h, download_dir=download_dir)
    try:
        a = U.start_app(tmp, root=root)
    finally:
        H.firefox = real
    try:
        yield a + (request.param,)
    finally:
        drv = a[0]
        try:
            drv.execute_script("window.__mace.settings.set('uiScale', 100)")
        except Exception:
            pass
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    app[0].execute_script("Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.plot.')).forEach((k) => localStorage.removeItem(k))")
    return app


def open_plot(drv, path):
    U.open_doc(drv, path, wait_atoms=False)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn === arguments[0]", 60, path)
    time.sleep(0.3)


# the subplots' data rectangles in viewport pixels: plotly's layout units times the UI scale
# (a CSS zoom on the shell), from the figure's own box
PLOTS_JS = """const PLOTS = (p, r) => { const L = p._fullLayout, z = r.width / L.width;
  return Object.keys(L).filter((k) => /^xaxis\\d*$/.test(k) && L[k]._length).map((k) => {
    const xa = L[k], ya = L['yaxis' + (xa.anchor || 'y').slice(1)] || L.yaxis;
    return [r.left + z * xa._offset, r.top + z * ya._offset, r.left + z * (xa._offset + xa._length), r.top + z * (ya._offset + ya._length)];
  }); };
"""

# every box in viewport pixels: the legends, the subplots' data rectangles, the tick labels
# and axis titles, the subtitle (and its line count)
BOXES = PLOTS_JS + """const p = document.querySelector('#plot-host .pv-plot'), r = p.getBoundingClientRect(), L = p._fullLayout;
  const box = (b) => [b.left, b.top, b.right, b.bottom];
  const legends = [...p.querySelectorAll('.infolayer > g')].filter((g) => /^legend\\d*$/.test(g.getAttribute('class'))).map((g) => {
    const bg = g.querySelector('rect.bg') || g;
    return {id: g.getAttribute('class'), box: box(bg.getBoundingClientRect()),
            texts: [...g.querySelectorAll('text[class^=legend][class$=text]')].map((t) => t.textContent)};
  }).filter((l) => l.box[2] - l.box[0] > 1);
  const plots = PLOTS(p, p.querySelector('.main-svg').getBoundingClientRect());
  const labels = [...p.querySelectorAll('.xtick text, .ytick text, .x2tick text, .g-xtitle text, .g-x2title text, .g-ytitle text')]
    .filter((t) => t.textContent.replace(/[\\s\\u00a0\\u200b]/g, '')).map((t) => ({t: t.textContent, box: box(t.getBoundingClientRect())}));
  const note = document.querySelector('#plot-host .pv-note'), nb = note.getBoundingClientRect();
  const lh = parseFloat(getComputedStyle(note).lineHeight) || 1.4 * parseFloat(getComputedStyle(note).fontSize);
  return {legends, plots, labels, note: box(nb), noteLines: Math.round(nb.height / lh), noteText: note.textContent,
          noteHelp: (note.querySelector('[data-help]') || {}).title || '', fig: box(r)};"""


def _hit(a, b, pad=0.0):
    return a[0] < b[2] - pad and b[0] < a[2] - pad and a[1] < b[3] - pad and b[1] < a[3] - pad


def _problems(st):
    out = []
    legs = st["legends"]
    for lg in legs:
        for k, pl in enumerate(st["plots"]):
            if _hit(lg["box"], pl, 1.5):
                out.append(f"legend {lg['texts']} {lg['box']} over plot area {k} {pl}")
        for lab in st["labels"]:
            if _hit(lg["box"], lab["box"], 1.5):
                out.append(f"legend {lg['texts']} over label {lab['t']!r}")
        if _hit(lg["box"], st["note"]):
            out.append(f"legend {lg['texts']} over the subtitle")
        f = st["fig"]
        if lg["box"][0] < f[0] - 1 or lg["box"][2] > f[2] + 1 or lg["box"][1] < f[1] - 1 or lg["box"][3] > f[3] + 1:
            out.append(f"legend {lg['texts']} cut by the figure edge")
    for i, a in enumerate(legs):
        for b in legs[i + 1:]:
            if _hit(a["box"], b["box"]):
                out.append(f"legends {a['texts']} and {b['texts']} overlap")
    return out


CASES = {
    # path, band legend texts, DOS legend texts (projections + spin key)
    "identical": ("gr/gra_band.BAND.DAT", ["α = β (non-magnetic)"], ["C (s)", "C (p)", "Total"], True),
    "magnetic": ("mag/fe_band.BAND.DAT", ["spin up (α)", "spin down (β)"], ["C (s)", "C (p)", "Total"], True),
    "closed": ("cl/mgo_band.BAND.DAT", [], MANY + ["Total"], False),
}


def _settle(drv):
    time.sleep(0.5)
    drv.execute_script("window.dispatchEvent(new Event('resize'))")
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn", 20)
    time.sleep(0.7)


@pytest.mark.parametrize("case", list(CASES))
def test_legends_stay_out_of_the_data_at_every_size(fresh, case):
    drv = fresh[0]
    path, band_names, dos_names, spin = CASES[case]
    open_plot(drv, path)
    failures = []
    for scale in (100, 110, 125):
        drv.execute_script("window.__mace.settings.set('uiScale', arguments[0])", scale)
        for width in (900, 1366, 1920):
            H.set_viewport(drv, width, 900)
            _settle(drv)
            st = drv.execute_script(BOXES)
            texts = [t for lg in st["legends"] for t in lg["texts"]]
            # each projection once; the spin direction is one shared key, only for a spin-polarized DOS
            for n in band_names + dos_names:
                if texts.count(n) != 1:
                    failures.append(f"{scale}% {width}px: {n!r} shown {texts.count(n)} times in {texts}")
            if any(t.endswith((" ↑", " ↓")) for t in texts):
                failures.append(f"{scale}% {width}px: a projection per spin in {texts}")
            keys = [t for t in texts if t.startswith(("↑", "↓"))]
            if spin and len(keys) != 2 or not spin and keys:
                failures.append(f"{scale}% {width}px: spin key {keys}")
            # the band legend and the DOS legend are separate legends, never one across both panels
            if band_names and dos_names:
                owner = {t: lg["id"] for lg in st["legends"] for t in lg["texts"]}
                if owner.get(band_names[0]) == owner.get(dos_names[0]):
                    failures.append(f"{scale}% {width}px: band and DOS entries share one legend")
            failures += [f"{scale}% {width}px: {p}" for p in _problems(st)]
            if width == 1366 and st["noteLines"] != 1:
                failures.append(f"{scale}% {width}px: the subtitle takes {st['noteLines']} lines: {st['noteText']!r}")
            if scale == 110 and width == 1366:
                U.shot(drv, f"legend_{case}_{drv.name}_110")
    drv.execute_script("window.__mace.settings.set('uiScale', 100)")
    H.set_viewport(drv, 1280, 800)
    assert not failures, "\n".join(failures)
    U.no_errors(drv)


def test_subtitle_is_one_short_line_with_the_rest_in_its_help(fresh):
    drv = fresh[0]
    open_plot(drv, "gr/gra_band.BAND.DAT")
    st = drv.execute_script(BOXES)
    note = st["noteText"]
    assert "spin-polarized" in note and "α = β (non-magnetic, ≤3 meV)" in note, note
    assert "Fermi level" in note and len(note) < 110, note
    # the longer guidance moved into the help icon's tooltip
    assert "↓ mirrored to the left" in st["noteHelp"] and "double-click to reset" in st["noteHelp"], st["noteHelp"]
    open_plot(drv, "mag/fe_band.BAND.DAT")
    assert "α solid, β dotted" in drv.execute_script(BOXES)["noteText"]
    U.no_errors(drv)


def test_png_and_svg_exports_keep_the_legends_outside(fresh):
    """Plotly's image export draws the same layout: its legends sit outside its plot areas."""
    drv = fresh[0]
    open_plot(drv, "gr/gra_band.BAND.DAT")
    st = drv.execute_async_script(PLOTS_JS + """const done = arguments[arguments.length - 1];
      const p = document.querySelector('#plot-host .pv-plot');
      Plotly.toImage(p, {format: 'svg', width: p.clientWidth, height: p.clientHeight}).then((url) => {
        const host = document.createElement('div');
        host.style.cssText = 'position:fixed;left:0;top:0;z-index:99999;background:#fff';
        host.innerHTML = decodeURIComponent(url.replace(/^data:image\\/svg\\+xml,/, ''));
        document.body.appendChild(host);
        const svg = host.querySelector('svg'), box = (b) => [b.left, b.top, b.right, b.bottom];
        const legends = [...svg.querySelectorAll('.infolayer > g')].filter((g) => /^legend\\d*$/.test(g.getAttribute('class'))).map((g) => ({box: box((g.querySelector('rect.bg') || g).getBoundingClientRect()),
          texts: [...g.querySelectorAll('text[class^=legend][class$=text]')].map((t) => t.textContent)}));
        const plots = PLOTS(p, svg.getBoundingClientRect());
        host.remove();
        done({legends, plots});
      });""")
    assert len(st["legends"]) == 2 and st["plots"], st
    bad = [(lg["texts"], pl) for lg in st["legends"] for pl in st["plots"] if _hit(lg["box"], pl, 1.5)]
    assert not bad, bad


@pytest.mark.skipif(not os.environ.get("MACE_LEGEND_SHOTS"), reason="screenshots on request (MACE_LEGEND_SHOTS)")
def test_screenshot_at_ui_scale_110(fresh):
    """The maintainer's view: graphite BAND + DOSS at UI scale 110 % in a 1600 x 960 window."""
    drv, browser = fresh[0], fresh[-1]
    H.set_viewport(drv, 1600, 960)
    drv.execute_script("window.__mace.settings.set('uiScale', 110)")
    for case in ("identical", "magnetic", "closed"):
        open_plot(drv, CASES[case][0])
        _settle(drv)
        target = Path(os.environ["MACE_LEGEND_SHOTS"])
        target.mkdir(parents=True, exist_ok=True)
        drv.save_screenshot(str(target / f"{os.environ.get('MACE_LEGEND_TAG', 'after')}_{case}_{browser}_110.png"))
    drv.execute_script("window.__mace.settings.set('uiScale', 100)")
    H.set_viewport(drv, 1280, 800)
