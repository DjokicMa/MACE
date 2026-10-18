"""The combined band + DOS plot of a spin-polarized run, asserted on the
plotly traces (no screenshots).

* "gr": a non-magnetic spin-polarized run in CRYSTAL's layout (graphite-like:
  beta = alpha to the last printed digit, a path break with no repeated x)
  with its DOSS partner (both spins, the edge pile-up CRYSTAL writes);
* "mag": a magnetic run (beta 54 meV off alpha);
* "slip": seekpath's oA2 path (Γ-Y-F₀|Δ₀-Γ-Z-B₀|G₀-T-Y|Γ-S-R-Z-T) with two
  very short segments, whose tick labels must not overlap.
"""
import sys
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
sys.path.insert(0, str(H.REPO_ROOT / "tests"))
from test_dos_spin_window import write_doss_spin  # noqa: E402
from test_gui_results import write_band_crystal, write_band_path  # noqa: E402


@pytest.fixture(scope="module", params=["firefox", "chromium"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bdspin")
    root = H.examples_copy(tmp)
    (root / "gr").mkdir()
    write_band_crystal(root / "gr", noise=1e-4)
    write_doss_spin(root / "gr")
    (root / "mag").mkdir()
    write_band_crystal(root / "mag", stem="fe_band", beta_shift=0.002)
    (root / "slip").mkdir()
    write_band_path(root / "slip")
    real = H.firefox
    if request.param == "chromium":     # start_app opens Firefox; the same app in Chromium
        H.firefox = lambda w, h, download_dir=None, **kw: H.chromium(w, h, download_dir=download_dir)
    try:
        a = U.start_app(tmp, root=root)
    finally:
        H.firefox = real
    try:
        yield a
    finally:
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


TRACES = """return document.querySelector('#plot-host .pv-plot').data.map((t) => ({
  name: t.name, meta: t.meta, fill: t.fill || 'none', pattern: t.fillpattern ? t.fillpattern.shape : null,
  showlegend: t.showlegend !== false, dash: t.line && t.line.dash || 'solid', xaxis: t.xaxis,
  nulls: (t.x || []).map((v, i) => v === null ? i : -1).filter((i) => i >= 0), x: t.x.slice(0, 12)}))"""


def test_non_magnetic_bands_once_with_a_gap_at_the_break(fresh):
    drv = fresh[0]
    open_plot(drv, "gr/gra_band.BAND.DAT")
    tr = drv.execute_script(TRACES)
    bands = [t for t in tr if t["meta"] in ("up", "down")]
    # one set, no spin legend (it is drawn in the plot's ink; the note says α = β)
    assert [t["name"] for t in bands] == ["α = β (non-magnetic)"] and not bands[0]["showlegend"]
    # 9 points per band, a null after band 1 and band 2, and one after the X|K tick (point 4)
    assert bands[0]["nulls"][:2] == [5, 10] and bands[0]["x"][4] == pytest.approx(1.0) and bands[0]["x"][6] == pytest.approx(1.25)
    note = drv.find_element("css selector", "#plot-host .pv-note").text
    assert "α = β (non-magnetic, ≤3 meV)" in note
    U.no_errors(drv)


def test_both_dos_spins_filled_and_in_the_legend(fresh):
    drv = fresh[0]
    open_plot(drv, "gr/gra_band.BAND.DAT")
    every = drv.execute_script(TRACES)
    tr = {t["name"]: t for t in every if t["meta"] == "dos"}
    # DOS colors Auto, one element: by orbital, named as the legend names them (1.1.6)
    assert {"Total ↑", "Total ↓", "C s ↑", "C s ↓", "C p ↑", "C p ↓"} <= set(tr)
    assert tr["Total ↑"]["fill"] == "tozerox" and tr["Total ↓"]["fill"] == "tozerox"
    assert tr["Total ↓"]["pattern"] is None and tr["Total ↑"]["pattern"] is None
    # the 1.1.6 legend: a Projections group of swatches (the total first) and neutral Spin keys
    assert not any(t["showlegend"] for t in tr.values())
    swatches = [t["name"] for t in every if t["meta"] == "key-proj"]
    assert swatches[0] == "Total" and set(swatches) == {"Total", "C s", "C p"}
    assert [(t["name"], t["dash"]) for t in every if t["meta"] == "key-spin"] == [("spin up (α)", "solid"), ("spin down (β)", "dash")]
    assert tr["Total ↓"]["dash"] == "dash" and tr["Total ↑"]["dash"] == "solid"
    for theme in ("dark", "light"):                  # each total filled lightly in its spin's color, in both themes
        drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
        time.sleep(0.5)
        fills = drv.execute_script("return document.querySelector('#plot-host .pv-plot').data"
                                   ".filter((t) => t.meta === 'dos' && t.fill === 'tozerox').map((t) => t.fillcolor)")
        assert len(fills) == 2 and fills[0] != fills[1] and all(f.startswith("rgba(") and f.endswith(",0.126)") for f in fills)
    U.no_errors(drv)


def test_magnetic_bands_keep_both_spins_beta_over_alpha(fresh):
    drv = fresh[0]
    open_plot(drv, "mag/fe_band.BAND.DAT")
    tr = [t for t in drv.execute_script(TRACES) if t["meta"] in ("up", "down")]
    # beta drawn last, dashed over alpha (1.1.6, as mace plotting): coinciding bands read in both colors
    assert [t["name"] for t in tr] == ["spin up (α)", "spin down (β)"]
    assert tr[0]["dash"] == "solid" and tr[1]["dash"] == "dash"
    note = drv.find_element("css selector", "#plot-host .pv-note").text
    assert "α solid, β dashed" in note
    U.no_errors(drv)


def _set_option(drv, ident, value):
    drv.execute_script("const i = document.getElementById(arguments[0]); i.value = arguments[1];"
                       " i.dispatchEvent(new Event('change', {bubbles: true}))", ident, str(value))


FIG = """const p = document.querySelector('#plot-host .pv-plot'), dos = p.data.filter((t) => t.meta === 'dos');
  const xs = dos.flatMap((t) => t.x.filter((v) => v !== null)), ys = dos.flatMap((t) => t.y.filter((v) => v !== null));
  return {x2: p.layout.xaxis2.range, y: p.layout.yaxis.range, xmax: Math.max(...xs.map(Math.abs)), ymin: Math.min(...ys), ymax: Math.max(...ys),
          bands: p.data.filter((t) => t.meta === 'up' || t.meta === 'down').length,
          shapes: p.layout.shapes.filter((s) => (s.name || '').startsWith('nodos')).map((s) => [s.name, s.y0, s.y1])}"""


def test_widening_the_energy_range_keeps_the_dos_scaled_to_its_data(fresh):
    """-10..+10 eV past a DOS window of about -7..+7 eV: no edge line, the DOS axis
    fits the data, the region without DOS is left blank (no gray band that reads as the total's fill,
    bug f) and the note says where the DOS was computed, the bands stay one set."""
    drv = fresh[0]
    open_plot(drv, "gr/gra_band.BAND.DAT")
    before = drv.execute_script(FIG)
    lo, hi = before["ymin"], before["ymax"]                 # the window (no Fermi shift, E_F = 0)
    assert lo == pytest.approx(-0.26 * 27.2114, abs=1e-3) and hi == pytest.approx(0.26 * 27.2114, abs=1e-3)
    _set_option(drv, "po-emin", -10)
    H.wait(drv, "return window.__mace.plotView.state().opts.emin === -10", 10)
    _set_option(drv, "po-emax", 10)
    H.wait(drv, "return window.__mace.plotView.state().opts.emax === 10 && window.__mace.plotView.state().drawn", 10)
    time.sleep(0.3)
    st = drv.execute_script(FIG)
    assert st["y"] == [-10, 10]
    assert st["xmax"] < 3                                    # 132 states/eV in the edge bin before
    assert max(map(abs, st["x2"])) < 3 * st["xmax"]          # the axis fits the data, not a spike
    assert st["bands"] == 1
    assert st["shapes"] == []
    tip = drv.find_element("css selector", "#plot-host .pv-note [data-help]").get_attribute("title")
    assert f"DOS computed from {lo:.2f} to {hi:.2f} eV" in tip, tip
    # zooming back in refits the DOS axis to the data in view
    drv.execute_script("return Plotly.relayout(document.querySelector('#plot-host .pv-plot'), {'yaxis.range': [-2, 2]})")
    time.sleep(0.5)
    assert max(map(abs, drv.execute_script(FIG)["x2"])) < 3
    U.no_errors(drv)


TICKS = """const out = [];
  document.querySelectorAll('#plot-host .pv-plot .subplot.xy .xaxislayer-above .xtick text').forEach((t) => {
    const lines = [...t.querySelectorAll('tspan.line')];
    const parts = (lines.length ? lines : [t]).filter((n) => n.textContent.replace(/[\\s\\u00a0\\u200b]/g, ""));
    const rects = parts.map((n) => { const r = n.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; });
    out.push({text: t.textContent, rects, sub: !!t.querySelector('tspan[dy]') || /baseline-shift|font-size/.test(t.innerHTML)});
  });
  const p = document.querySelector('#plot-host .pv-plot');
  return {labels: out, angle: p.layout.xaxis.tickangle, b: p.layout.margin.b, text: p.layout.xaxis.ticktext};"""


def _overlapping(labels):
    """Labels whose line boxes overlap. A line box is the font's full line (ascent to
    descent, about 19 px at 12 px): rows one line apart share about 2 px of it with no
    ink there, so a vertical overlap counts from 4 px."""
    boxes = [(lab["text"], r) for lab in labels for r in lab["rects"]]
    return [(a[0], b[0]) for i, a in enumerate(boxes) for b in boxes[i + 1:] if a[0] != b[0]
            and a[1][0] < b[1][2] - 0.5 and b[1][0] < a[1][2] - 0.5 and a[1][1] < b[1][3] - 4 and b[1][1] < a[1][3] - 4]


def test_crowded_k_labels_drop_a_row_instead_of_overlapping(fresh):
    drv = fresh[0]
    open_plot(drv, "slip/slip_band.BAND.DAT")
    st = drv.execute_script(TICKS)
    texts = [lab["text"].replace("\u200b", "").replace("\xa0", "") for lab in st["labels"]]   # plotly pads subscripts with zero-width spaces
    assert texts == ["Γ", "Y", "F0|Δ0", "Γ", "Z", "B0|G0", "T", "Y|Γ", "S", "R", "Z", "T"], texts
    assert st["angle"] == 0 and not _overlapping(st["labels"]), _overlapping(st["labels"])
    assert any("<br>" in t for t in st["text"]) and st["b"] > 42      # a second row, room for it
    assert all(lab["sub"] for lab in st["labels"] if "0" in lab["text"])       # F₀, Δ₀, B₀, G₀ as subscripts
    # the band lines break at all three discontinuities
    tr = [t for t in drv.execute_script(TRACES) if t["meta"] == "up"]
    assert len(tr) == 1
    # zoomed in on Y-F0 the labels have room again: one row
    drv.execute_script("return Plotly.relayout(document.querySelector('#plot-host .pv-plot'), {'xaxis.range': [0.45, 0.65]})")
    H.wait(drv, "return document.querySelector('#plot-host .pv-plot').layout.margin.b === 42", 10)
    st = drv.execute_script(TICKS)
    assert not _overlapping(st["labels"]) and not any("<br>" in t for t in st["text"])
    for width in (900, 1600):                        # narrow and wide windows
        H.set_viewport(drv, width, 800)
        time.sleep(0.8)
        drv.execute_script("return Plotly.relayout(document.querySelector('#plot-host .pv-plot'), {'xaxis.range': [0, 3.85746]})")
        time.sleep(0.6)
        assert not _overlapping(drv.execute_script(TICKS)["labels"]), width
    H.set_viewport(drv, 1280, 800)
    U.no_errors(drv)
