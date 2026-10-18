"""The band/DOS plot draws the API's energies (CLI/GUI parity audit, 1.1.6).

The page computes nothing the bands/dos API (results.py, the readers of
`mace plotting`) has already decided; it only applies the reader's own Fermi
shift. Three places did their own arithmetic:

* with a POTC run beside the data the API (and `mace plotting`) give the
  energies on the vacuum scale, with E_F at the vacuum shift; the page
  subtracted E_F again, so it drew E - E_F under an "E - E_vac" axis;
* the DOS axis was fitted to the states of the window without the Fermi
  shift, while the curves are drawn shifted;
* on a DOS-only plot the dashed Fermi line could be dragged but the drag set
  no shift.

Synthetic BAND.DAT / DOSS.DAT / POTC files (the writers of test_gui_results
and test_ipdos_readers), as test_plot_e2e."""
import sys
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
sys.path.insert(0, str(H.REPO_ROOT / "tests"))
from test_gui_results import write_band  # noqa: E402
from test_ipdos_readers import write_doss  # noqa: E402

PLOT = "document.querySelector('#plot-host .pv-plot')"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("plotmath")
    root = H.examples_copy(tmp)
    for d in ("results", "vac", "dosonly"):
        (root / d).mkdir()
    write_band(root / "results")
    write_doss(root / "results", material="mgo")
    write_band(root / "vac")
    write_doss(root / "vac", material="mgo")
    # a POTC run beside them: the vacuum level (test_gui_results' numbers)
    (root / "vac" / "mgo_POTC.POTC.DAT").write_text("# z V\n0.0 0.1\n1.0 0.1\n")
    (root / "vac" / "mgo_POTC.out").write_text(" FERMI ENERGY      -0.2\n")
    write_doss(root / "dosonly", material="zno")
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


def open_plot(drv, path):
    U.open_doc(drv, path, wait_atoms=False)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn === arguments[0]", 60, path)
    time.sleep(0.3)


def _drag_fermi_to(drv, y):
    drv.execute_script(f"return Plotly.relayout({PLOT}, {{'shapes[0].y0': arguments[0], 'shapes[0].y1': arguments[0]}})", y)


def test_vacuum_scale_bands_are_the_apis_energies(fresh):
    drv, base = fresh[0], fresh[1]
    st, _, api = U.req(base, "api/bands?path=vac/mgo_band.BAND.DAT")
    assert st == 200 and api["reference"] == "vacuum" and abs(api["fermi"]) > 1
    open_plot(drv, "vac/mgo_band.BAND.DAT")
    fig = drv.execute_script(f"const p = {PLOT}; return {{traces: p.data.filter((t) => t.meta === 'up').map((t) => t.y),"
                             " shapes: p.layout.shapes.filter((s) => s.name === 'fermi').map((s) => s.y0),"
                             " range: p.layout.yaxis.range, title: p.layout.yaxis.title.text}")
    drawn = [y for y in fig["traces"][0] if y is not None]
    want = [e for band in api["up"] for e in band if e is not None]
    assert drawn == pytest.approx(want, abs=1e-9)                  # the API's (and mace plotting's) energies
    assert fig["shapes"] == [pytest.approx(api["fermi"])]           # E_F at the vacuum shift, as the CLI draws it
    assert fig["range"] == pytest.approx([api["fermi"] - 6, api["fermi"] + 6])    # the window is around E_F
    assert "vac" in fig["title"]
    # a drag of the line still shifts the reference by where it went
    _drag_fermi_to(drv, api["fermi"] + 0.5)
    H.wait(drv, "return Math.abs(window.__mace.plotView.state().opts.shift - 0.5) < 1e-6", 10)
    U.no_errors(drv)


def test_the_dos_axis_fits_the_shifted_window(fresh):
    drv = fresh[0]
    open_plot(drv, "results/mgo_band.BAND.DAT")
    # +6 eV: the states drawn are those from E_F to E_F + 12 eV, whose largest value is not the window's
    _drag_fermi_to(drv, 6.0)
    H.wait(drv, "return Math.abs(window.__mace.plotView.state().opts.shift - 6) < 1e-6", 10)
    H.wait(drv, f"const p = {PLOT}; return p.layout.xaxis2 && p.data.some((t) => t.meta === 'dos')", 10)
    time.sleep(0.5)
    got = drv.execute_script(f"""const p = {PLOT}, [lo, hi] = p.layout.yaxis.range;
        let m = 0;
        p.data.filter((t) => t.meta === 'dos').forEach((t) => t.y.forEach((y, i) => {{
          if (y >= Math.min(lo, hi) && y <= Math.max(lo, hi)) m = Math.max(m, Math.abs(t.x[i] || 0)); }}));
        return {{range: p.layout.xaxis2.range, drawnMax: m}}""")
    # the axis reaches 8 % past the largest DOS value drawn in the visible energies
    assert max(abs(v) for v in got["range"]) == pytest.approx(got["drawnMax"] * 1.08, rel=1e-6), got


def test_a_dos_only_plot_takes_the_fermi_drag(fresh):
    drv = fresh[0]
    open_plot(drv, "dosonly/zno_doss.DOSS.DAT")
    y0 = drv.execute_script(f"return {PLOT}.layout.shapes[0].y0")
    _drag_fermi_to(drv, y0 + 0.75)
    H.wait(drv, "return Math.abs(window.__mace.plotView.state().opts.shift - 0.75) < 1e-6", 10)
    U.no_errors(drv)


def _norm_svg(text):
    """matplotlib's and plotly's run-to-run noise (the date, clip, marker and plot ids) taken out."""
    import re
    t = re.sub(r"<dc:date>.*?</dc:date>", "", text)
    uid = re.search(r'id="clip([0-9a-f]{6})', t)            # plotly's figure id, in every clip path and def
    if uid:
        t = t.replace(uid.group(1), "UID")
    t = re.sub(r"\btrace[0-9a-f]{6}\b", "traceID", t)
    ids = {}
    return re.sub(r"\b[pm][0-9a-f]{10}\b", lambda m: ids.setdefault(m.group(0), f"ID{len(ids)}"), t)


@pytest.mark.parametrize("rel, ref", [("vac/mgo_band.BAND.DAT", "vacuum"), ("dosonly/zno_doss.DOSS.DAT", None)])
def test_publication_figure_takes_the_fermi_shift_and_reference(fresh, rel, ref):
    """Publication Figure… passes the shift on screen (--fermi-shift) and, with a POTC run, the vacuum
    scale the page draws (--energy-reference vacuum; the POTC files go along to the cache folder): the
    figure is the one its shown command draws in the data's own folder, on the same energy zero."""
    import shlex
    import shutil
    import subprocess
    import urllib.parse
    drv, base, root = fresh[0], fresh[1], fresh[2]
    open_plot(drv, rel)
    y0 = drv.execute_script(f"return {PLOT}.layout.shapes[0].y0")
    _drag_fermi_to(drv, y0 + 0.5)
    H.wait(drv, "return Math.abs(window.__mace.plotView.state().opts.shift - 0.5) < 1e-6", 10)
    r = drv.execute_async_script("const done = arguments[0]; window.__mace.plotView.publication().then(done, (e) => done({error: String(e)}))")
    assert "error" not in r, r
    cmd = r["command"]
    assert "--fermi-shift=0.5" in cmd, cmd
    assert ("--energy-reference=vacuum" in cmd) == (ref == "vacuum"), cmd
    assert "Error" not in r.get("log", ""), r.get("log")
    svg = next(n for n in r["images"] if n.endswith(".svg"))
    st, _, served = U.req(base, "api/plot-image?" + urllib.parse.urlencode({"key": r["key"], "name": svg}))
    assert st == 200
    served = served.decode() if isinstance(served, bytes) else served
    # mace plotting draws the GUI's figure (plotly, 1.1.6): with the shift the energy axis reads E − (E_vac + s)
    # or E − (E_F + s) and the dashed line, which stays put, is E_F + s on either scale (1.1.6 review F4-2)
    import re as _re
    text = _re.sub(r"<[^>]+>", "", served).replace("\u200b", "")
    zero = "Evac" if ref == "vacuum" else "EF"
    assert f"E − ({zero} + 0.50 eV) (eV)" in text and "EF + 0.50 eV" in text, text[-400:]
    # the shown command, run by hand in a copy of the data's folder, draws the same figure
    folder = root / rel.split("/")[0]
    by_hand = fresh[3].parent / f"by_hand_{folder.name}"
    shutil.rmtree(by_hand, ignore_errors=True)
    shutil.copytree(folder, by_hand)
    p = subprocess.run([sys.executable, str(H.REPO_ROOT / "mace_cli"), "--no-banner", *shlex.split(cmd)[1:]],
                       cwd=str(by_hand), capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert _norm_svg((by_hand / svg).read_text()) == _norm_svg(served)
    U.no_errors(drv)
